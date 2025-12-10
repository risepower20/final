# LFM2-8B GPU 추론 파이프라인 개발 로그

---

## 전체 추론 파이프라인 의사코드

```
ALGORITHM LFM2_Inference(input_ids, model_weights)
  INPUT:
    - input_ids: int32 배열 [batch_size × seq_len]
    - model_weights: 사전 로드된 GPU 가중치
  
  OUTPUT:
    - logits: float32 배열 [batch_size × seq_len × vocab_size]
  
  BEGIN
    // 1. Embedding Lookup
    hidden_states ← EmbeddingLookup(input_ids, embed_table)
    // hidden_states: [batch × seq × hidden_size]
    
    // 2. Multi-GPU Pipeline (4 GPUs, 6 layers each)
    FOR stage = 0 TO 3 DO
      SetDevice(GPU[stage])
      
      // Double Stream으로 compute/transfer 오버랩
      FOR layer = stage*6 TO stage*6+5 DO
        
        // 2.1 Pre-normalization
        normed ← RMSNorm(hidden_states)
        
        // 2.2 Attention 또는 Conv 분기
        IF LAYER_TYPES[layer] == 0 THEN  // Attention
          // Q, K, V projection (Tiled GEMM)
          Q ← TiledGEMM(normed, W_q)  // [batch × seq × num_heads × head_dim]
          K ← TiledGEMM(normed, W_k)  // [batch × seq × num_kv_heads × head_dim]
          V ← TiledGEMM(normed, W_v)
          
          // RoPE 적용
          Q, K ← ApplyRoPE(Q, K, position)
          
          // Fused GQA Attention
          attn_out ← FusedGQA(Q, K, V, causal_mask)
          
          // Output projection
          layer_out ← TiledGEMM(attn_out, W_o)
          
        ELSE  // Conv
          // Causal Conv1D
          layer_out ← CausalConv1D(normed, conv_weight, conv_cache)
        END IF
        
        // 2.3 Residual connection
        hidden_states ← hidden_states + layer_out
        
        // 2.4 Post-normalization
        normed ← RMSNorm(hidden_states)
        
        // 2.5 MoE Feed-Forward
        // Router: expert 선택
        router_logits ← GEMM(normed, router_weight)
        top_k_experts, weights ← TopK(Softmax(router_logits), k=4)
        
        // Persistent MoE Kernel (동적 로드 밸런싱)
        moe_out ← PersistentMoE(normed, top_k_experts, weights, expert_weights)
        
        // Residual
        hidden_states ← hidden_states + moe_out
      END FOR
      
      // GPU 간 데이터 전송 (P2P)
      IF stage < 3 THEN
        EventRecord(transfer_event, compute_stream)
        StreamWaitEvent(next_gpu_stream, transfer_event)
        P2PTransfer(hidden_states, GPU[stage] → GPU[stage+1])
      END IF
    END FOR
    
    // 3. Final Normalization
    hidden_states ← RMSNorm(hidden_states)
    
    // 4. LM Head (vocabulary projection)
    logits ← LMHeadGEMM(hidden_states, lm_head_weight)
    
    RETURN logits
  END
```

---

## 입력/출력 데이터 명세

### 입력 데이터
| 항목 | 형식 | 범위/크기 | 설명 |
|------|------|-----------|------|
| input_ids | int32 | [0, 65535] | 토큰 ID (VOCAB_SIZE=65536) |
| batch_size | int | 1 ~ 384 | OOM 방지 상한 |
| seq_len | int | 1 ~ 128000 | MAX_POSITION_EMBEDDINGS |
| BUFFER_SEQ_LEN | int | 32 (고정) | 파이프라인 버퍼 크기 |

### 출력 데이터
| 항목 | 형식 | 크기 | 설명 |
|------|------|------|------|
| logits | float32 | [batch × seq × 65536] | 각 토큰별 vocabulary logits |
| hidden_states | float32 | [batch × seq × 2048] | 중간 hidden states |

### 가중치 데이터
| 항목 | 형식 | 크기 | 설명 |
|------|------|------|------|
| embed_tokens | float32 | [65536 × 2048] | 임베딩 테이블 |
| layer weights | float32 | 레이어별 상이 | Q/K/V/O proj, MLP, MoE 등 |
| lm_head | float32 | [2048 × 65536] | 출력 projection |

### 메모리 요구사항
| GPU | 레이어 | 가중치 크기 | 버퍼 크기 | 총 사용량 |
|-----|--------|------------|----------|----------|
| GPU 0 | 0-5 | ~3.2GB | ~0.5GB | ~3.7GB |
| GPU 1 | 6-11 | ~3.2GB | ~0.5GB | ~3.7GB |
| GPU 2 | 12-17 | ~3.2GB | ~0.5GB | ~3.7GB |
| GPU 3 | 18-23 | ~3.2GB | ~0.5GB | ~3.7GB |

### 수치 정밀도
- 내부 연산: FP32 (float)
- FP16 옵션: `--fp16` 플래그로 활성화 가능
- Epsilon 값: RMS_NORM_EPS = 1e-5

### 재현성 보장
- 동일 입력 → 동일 출력 (deterministic)
- 검증: 1024/1024 샘플 일치
- max_diff < 1e-5 (수치 오차 범위)

---

12/01 (일): 프로젝트 초기 설정

먼저 모델 파라미터를 정의했다.

```cpp
// include/config.h
#pragma once
#include <cstddef>

// Model parameters - LFM2-8B 스펙에 맞춤
constexpr size_t VOCAB_SIZE = 65536;
constexpr size_t HIDDEN_SIZE = 2048;
constexpr size_t INTERMEDIATE_SIZE = 7168;
constexpr size_t NUM_HIDDEN_LAYERS = 24;
constexpr size_t NUM_ATTENTION_HEADS = 32;
constexpr size_t NUM_KEY_VALUE_HEADS = 8;  // GQA
constexpr size_t HEAD_DIM = HIDDEN_SIZE / NUM_ATTENTION_HEADS;  // 64
```

GQA(Grouped Query Attention)를 사용하기 때문에 KV heads가 8개뿐이다. Query heads 32개가 KV heads 8개를 공유한다.

MoE 파라미터도 추가:

```cpp
// MoE parameters
constexpr size_t NUM_EXPERTS = 32;
constexpr size_t NUM_EXPERTS_PER_TOK = 4;  // Top-4
constexpr size_t MOE_INTERMEDIATE_SIZE = 1792;
```

레이어 타입 배열 정의. 0=Attention, 1=Conv:

```cpp
constexpr int LAYER_TYPES[] = {
    1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1,
    1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1
};
```

24개 레이어 중 Attention이 6개(layer 2,6,10,14,18,21), 나머지 18개는 Conv.

---

Tensor 클래스 인터페이스 설계:

```cpp
// include/tensor.h
class Tensor {
public:
    Tensor();
    Tensor(const std::vector<size_t>& shape);
    Tensor(const std::vector<size_t>& shape, float* data, bool copy = true);
    ~Tensor();
    
    // Copy/Move
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    
    // Shape
    size_t ndim() const { return shape_.size(); }
    const std::vector<size_t>& shape() const { return shape_; }
    size_t size() const { return size_; }
    
    // Data access
    float* data() { return data_; }
    float& at(size_t i);
    float& at(size_t i, size_t j);
    float& at(size_t i, size_t j, size_t k);
    float& at(size_t i, size_t j, size_t k, size_t l);

private:
    std::vector<size_t> shape_;
    size_t size_;
    float* data_;
    bool owns_data_;
};
```

핵심은 `owns_data_` 플래그다. true면 메모리를 직접 관리하고, false면 다른 Tensor의 메모리를 참조만 한다 (view).

---

12/03 (수): Tensor, 모델 로더 구현 및 헤더 정의

생성자 구현

```cpp
// src/tensor.cu
Tensor::Tensor() : size_(0), data_(nullptr), owns_data_(false) {}

Tensor::Tensor(const std::vector<size_t>& shape)
    : shape_(shape), owns_data_(true) {
    size_ = compute_size();
    allocate();
}

Tensor::Tensor(const std::vector<size_t>& shape, float* data, bool copy)
    : shape_(shape), owns_data_(copy) {
    size_ = compute_size();
    if (copy) {
        allocate();
        std::memcpy(data_, data, size_ * sizeof(float));
    } else {
        data_ = data;  // view
    }
}
```

Move semantics 구현

성능을 위해 move가 중요하다:

```cpp
Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)),
      size_(other.size_),
      data_(other.data_),
      owns_data_(other.owns_data_) {
    // 원본 무효화
    other.data_ = nullptr;
    other.size_ = 0;
    other.owns_data_ = false;
}
```

Move하면 데이터 복사 없이 포인터만 넘어간다.

ModelLoader 구현

모델 가중치를 파일에서 읽어오는 클래스:

```cpp
// include/model_loader.h
class ModelLoader {
public:
    ModelLoader(const std::string& model_path);
    ~ModelLoader();
    
    Tensor load_tensor(const std::string& name);
    bool has_tensor(const std::string& name) const;
    
private:
    std::string model_path_;
    std::ifstream file_;
    
    struct TensorInfo {
        size_t offset;
        size_t size;
        std::vector<size_t> shape;
    };
    
    std::unordered_map<std::string, TensorInfo> tensor_index_;
    
    void build_index();
};
```

핵심은 텐서 인덱스를 미리 빌드해서 필요할 때 offset으로 바로 접근하는 것.

```cpp
// src/model_loader.cpp
void ModelLoader::build_index() {
    // 파일 헤더 파싱하여 각 텐서의 위치/크기/shape 기록
    while (file_) {
        std::string name;
        std::vector<size_t> shape;
        size_t offset;
        // ... 파싱 ...
        tensor_index_[name] = {offset, size, shape};
    }
}

Tensor ModelLoader::load_tensor(const std::string& name) {
    auto it = tensor_index_.find(name);
    if (it == tensor_index_.end()) {
        throw std::runtime_error("Tensor not found: " + name);
    }
    
    file_.seekg(it->second.offset);
    Tensor t(it->second.shape);
    file_.read(reinterpret_cast<char*>(t.data()), 
               it->second.size * sizeof(float));
    return t;
}
```

처음엔 seekg 없이 순차적으로 읽어서 특정 텐서만 로드할 수 없었다. 인덱스 방식으로 변경.

다차원 인덱싱

Row-major layout 가정:

```cpp
float& Tensor::at(size_t i, size_t j) {
    return data_[i * shape_[1] + j];
}

float& Tensor::at(size_t i, size_t j, size_t k) {
    return data_[(i * shape_[1] + j) * shape_[2] + k];
}

float& Tensor::at(size_t i, size_t j, size_t k, size_t l) {
    return data_[((i * shape_[1] + j) * shape_[2] + k) * shape_[3] + l];
}
```

테스트

```
$ make clean && make
$ ./main --benchmark
# Tensor 로드 테스트
Loading model weights... OK
Tensor creation: PASS
Tensor copy: PASS
Tensor move: PASS
Tensor indexing: PASS
All tests passed!
```

---

12/03 (수) 계속: 레이어 인터페이스 설계

각 레이어 타입별 클래스 인터페이스 정의:

```cpp
// include/layer.h
class RMSNorm {
public:
    RMSNorm(size_t hidden_size, float eps = RMS_NORM_EPS);
    void load_weights(const std::string& prefix);
    Tensor forward(const Tensor& x);
private:
    Tensor weight_;
    float eps_;
};

class Attention {
public:
    Attention(size_t hidden_size, size_t num_heads, size_t num_kv_heads);
    void load_weights(const std::string& prefix);
    Tensor forward(const Tensor& x, size_t position);
private:
    Tensor q_proj_, k_proj_, v_proj_, o_proj_;
    std::unique_ptr<RotaryEmbedding> rope_;
    size_t num_heads_, num_kv_heads_, head_dim_;
};

class CausalConv1D {
public:
    CausalConv1D(size_t channels, size_t kernel_size);
    Tensor forward(const Tensor& x);
private:
    Tensor weight_, bias_;
    Tensor cache_;  // Conv state cache
};

class MoE {
public:
    MoE(size_t hidden_size, size_t num_experts, size_t top_k);
    Tensor forward(const Tensor& x);
private:
    Tensor router_weight_;
    std::vector<MLP> experts_;
};

class TransformerBlock {
public:
    TransformerBlock(int layer_idx, int layer_type);
    Tensor forward(const Tensor& x, size_t position);
private:
    int layer_type_;  // 0=Attention, 1=Conv
    std::unique_ptr<Attention> attention_;
    std::unique_ptr<CausalConv1D> conv_;
    std::unique_ptr<MoE> moe_;
};
```

layer_type으로 Attention/Conv 분기 처리.

전체 모델 클래스:

```cpp
// include/model.h
class LFM2Model {
public:
    LFM2Model();
    ~LFM2Model();
    
    void load_weights(const std::string& model_path);
    Tensor forward(const Tensor& input_ids);
    Tensor generate(const Tensor& input_ids, size_t max_new_tokens);
    
private:
    Tensor embed_tokens_;
    std::vector<std::unique_ptr<TransformerBlock>> layers_;
    std::unique_ptr<RMSNorm> final_norm_;
    Tensor lm_head_;
    
    // KV Cache
    struct KVCache {
        Tensor key;
        Tensor value;
    };
    std::vector<KVCache> kv_cache_;
};
```

KVCache를 레이어별로 관리.

---

12/05 (금): RMSNorm, 기본 GEMM 커널 구현

성능 최적화 과정

첫 번째 시도: Naive GEMM

```cuda
__global__ void gemm_naive(float* A, float* B, float* C,
                           int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

결과: 너무 느림. Global memory 접근이 병목.

```
$ make clean && make
$ ./main --benchmark
Throughput: 0.3 samples/sec
```

충격적으로 느림. 1 sample/sec도 안 나옴.

Nsight Compute 분석 결과:
```
Memory Throughput: 23.4%
Compute Throughput: 12.1%
L2 Cache Hit Rate: 8.2%
Global Load Efficiency: 25%
```

문제점:
- 각 스레드가 K번 global memory 접근 (K=2048)
- Coalesced access 아님 (stride 접근 패턴)
- 같은 데이터를 여러 스레드가 중복 로드
- Memory bound: 연산보다 메모리 접근이 병목

두 번째 시도: Tiled GEMM (32x32)

Shared memory 활용:

```cuda
__global__ void gemm_tiled_v1(float* A, float* B, float* C,
                              int M, int N, int K) {
    __shared__ float As[32][8];
    __shared__ float Bs[8][32];
    
    int row = blockIdx.y * 32 + threadIdx.y;
    int col = blockIdx.x * 32 + threadIdx.x;
    float sum = 0.0f;
    
    for (int t = 0; t < K; t += 8) {
        // Load tiles to shared memory
        if (threadIdx.x < 8)
            As[threadIdx.y][threadIdx.x] = A[row * K + t + threadIdx.x];
        if (threadIdx.y < 8)
            Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
        __syncthreads();
        
        // Compute
        for (int k = 0; k < 8; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    
    C[row * N + col] = sum;
}
```

결과: 개선됨!

```
$ make clean && make
$ ./main --benchmark
Throughput: 12.4 samples/sec
```

Naive 대비 40배 향상. 하지만 inner loop가 K/8=256번 돌아서 오버헤드가 있다.

Nsight Compute 분석:
```
Memory Throughput: 58.3%
Compute Throughput: 45.2%
L2 Cache Hit Rate: 67.4%
Shared Memory Efficiency: 72%
Achieved Occupancy: 50%
```

개선점:
- Global memory 접근 횟수 8배 감소
- Shared memory에서 데이터 재사용
- L2 캐시 히트율 크게 향상

문제점:
- 여전히 Compute throughput이 낮음
- 타일이 작아서 shared memory 재사용률 부족
- inner loop 오버헤드 (K/8=256번 반복)

세 번째 시도: 타일 크기 증가 (64x64x32)

```cpp
// config.h에 추가
constexpr int GEMM_TILE_M = 64;
constexpr int GEMM_TILE_N = 64;
constexpr int GEMM_TILE_K = 32;
```

Shared Memory를 활용한 블록 단위 계산으로 Global Memory 접근 최소화.

```
# config.h 수정: TILE 64x64x32
$ make clean && make
$ ./main --benchmark
Throughput: 78.3 samples/sec
```

Tiled v1 대비 6배 향상!

Nsight Compute 분석:
```
Memory Throughput: 71.2%
Compute Throughput: 62.8%
Shared Memory Efficiency: 85%
Achieved Occupancy: 62.5%
Warp Execution Efficiency: 94%
```

타일 크기 실험 (실패 케이스들):

시도 3-1: TILE 256x256x32
```
# config.h: GEMM_TILE_M=256, GEMM_TILE_N=256, GEMM_TILE_K=32
$ make clean && make
$ ./main --benchmark
CUDA error: too many resources requested for launch
```
Shared memory 초과. 256*256*4 + 32*256*4 = 294912 bytes > 48KB 제한.

시도 3-2: TILE 128x64x16
```
# config.h: GEMM_TILE_M=128, GEMM_TILE_N=64, GEMM_TILE_K=16
$ make clean && make
$ ./main --benchmark
Throughput: 65.2 samples/sec
```
성능 오히려 감소. 타일이 비대칭이라 load balance 안 맞음.

시도 3-3: TILE 96x96x24
```
# config.h: GEMM_TILE_M=96, GEMM_TILE_N=96, GEMM_TILE_K=24
$ make clean && make
$ ./main --benchmark
Throughput: 71.8 samples/sec
```
96이 32의 배수가 아니라서 warp 효율 떨어짐.

시도 3-4: TILE 64x64x64
```
# config.h: GEMM_TILE_M=64, GEMM_TILE_N=64, GEMM_TILE_K=64
$ make clean && make
$ ./main --benchmark
CUDA error: too many resources requested for launch
```
K 타일이 너무 커서 shared memory 초과.

시도 3-5: TILE 64x64x16
```
# config.h: GEMM_TILE_M=64, GEMM_TILE_N=64, GEMM_TILE_K=16
$ make clean && make
$ ./main --benchmark
Throughput: 82.1 samples/sec
```
K=16이 작아서 inner loop 오버헤드 많음. 64x64x32 대비 소폭 향상.

최종 선택: TILE 128x128x16 (Attention용)

```
# config.h: ATTN_TILE_M=128, ATTN_TILE_N=128, ATTN_TILE_K=16
$ make clean && make
$ ./main --benchmark
Throughput: 178 samples/sec
```

참고: 128x128x16은 double buffering + padding으로 shared memory 사용량 최적화하여 가능.

네 번째 시도: Float4 벡터 로드

한 번에 4개의 float을 읽어 메모리 대역폭 효율 향상:

```cuda
// 기존: float 하나씩
float a = A[idx];

// 개선: float4로 한번에 4개
float4 a4 = reinterpret_cast<float4*>(A)[idx/4];
```

처음엔 alignment 에러 발생:
```
CUDA error: misaligned address
```
주소가 16바이트 정렬되어야 함. 배열 할당 시 alignment 보장하도록 수정.

```
# float4 로드 적용
$ make clean && make
$ ./main --benchmark
Throughput: 256 samples/sec
```

다섯 번째 시도: Double Buffering

다음 타일 데이터를 미리 로드하면서 현재 타일 연산 수행:

```cuda
// Prefetch next tile while computing current
__shared__ float As[2][TILE][TILE];  // 더블 버퍼
int curr = 0, next = 1;

for (int t = 0; t < K; t += TILE_K) {
    // Load next tile (async)
    load_tile_async(As[next], A, t + TILE_K);
    
    // Compute current tile
    compute_tile(As[curr], Bs[curr], acc);
    
    swap(curr, next);
    __syncthreads();
}
```

메모리 로드와 연산이 겹쳐서 파이프라인 효율 향상.

```
# double buffering 적용
$ make clean && make
$ ./main --benchmark
Throughput: 312 samples/sec
```

여섯 번째 시도: Register Tiling

각 스레드가 여러 output element를 계산:

```cuda
float acc[4][4];  // 각 스레드가 4x4 output 담당
```

```
# register tiling 적용
$ make clean && make
$ ./main --benchmark
Throughput: 358 samples/sec
```

GEMM 최적화 요약:
```
Naive:          0.3 samples/sec (baseline)
Tiled 32x32:    12.4 samples/sec (41x)
Tiled 64x64:    78.3 samples/sec (261x)
+ 128x128:      178 samples/sec (593x)
+ Float4:       256 samples/sec (853x)
+ Double buf:   312 samples/sec (1040x)
+ Reg tile:     358 samples/sec (1193x)
```

최종 GEMM 커널 (attn_matmul_tiled_kernel)

실제 구현된 최적화 GEMM:

```cuda
// src/layer.cu
__global__ void attn_matmul_tiled_kernel(float* out, const float* A,
                                         const float* B, int M, int N, int K) {
  const int BM = 128, BN = 128, BK = 16;  // ATTN_TILE_M, ATTN_TILE_N, ATTN_TILE_K
  const int TM = 8, TN = 8;  // ATTN_THREAD_TILE_M, ATTN_THREAD_TILE_N
  
  // Double buffer with padding (ATTN_PADDING=4)
  __shared__ float As[2][BM][BK + 4];
  __shared__ float Bs[2][BK][BN + 4];
  
  float threadResults[TM * TN] = {0.0f};
  float regM[TM], regN[TN];
  
  // Vectorized load with float4
  const float4* A_vec = reinterpret_cast<const float4*>(A);
  const float4* B_vec = reinterpret_cast<const float4*>(B);
  
  // Prefetch first tile
  for (int i = 0; i < 2; ++i) {
    int r = load_a_row + i * 64;
    float4 loaded_a = A_vec[(global_row_a * K + global_col_a) / 4];
    As[0][r][load_a_col + 0] = loaded_a.x;
    As[0][r][load_a_col + 1] = loaded_a.y;
    As[0][r][load_a_col + 2] = loaded_a.z;
    As[0][r][load_a_col + 3] = loaded_a.w;
  }
  
  // Main loop with double buffering
  for (int k = 0; k < K; k += BK) {
    int cur_buf = (k / BK) % 2;
    int nxt_buf = ((k / BK) + 1) % 2;
    
    // Async load next tile while computing current
    if (next_k < K) {
      // Load to nxt_buf...
    }
    
    // Compute using cur_buf
    for (int kk = 0; kk < BK; ++kk) {
      // Register-level tiling
      for (int m = 0; m < TM; ++m) regM[m] = As[cur_buf][...][kk];
      for (int n = 0; n < TN; ++n) regN[n] = Bs[cur_buf][kk][...];
      for (int m = 0; m < TM; ++m)
        for (int n = 0; n < TN; ++n)
          threadResults[m * TN + n] += regM[m] * regN[n];
    }
    __syncthreads();
  }
}
```

핵심 최적화:
- Double buffering: 메모리 로드와 연산 오버랩
- Float4 vectorized load: 한 번에 128bit 읽기
- Register-level tiling: 각 스레드가 TM x TN 결과 계산
- Shared memory padding: Bank conflict 방지

---

12/07 (일): Attention, Conv, MoE 커널 완성

Attention 레이어에서 segfault:

```
$ ./main --benchmark
Running layer 2 (Attention)...
Segmentation fault (core dumped)
```

gdb로 확인:

```
Program received signal SIGSEGV
0x00007f... in cudaMalloc ()
```

알고 보니 shape이 (0, 32, 64)였다. 입력 텐서 초기화 안 함.

그 다음:

```
$ ./main --benchmark
Running layer 2 (Attention)...
CUDA error: misaligned address
  at apply_rotary_pos_emb
```

float4 로드인데 주소가 16바이트 정렬 안 됨. reinterpret_cast 잘못 씀.

```cpp
// 버그
float4* ptr = reinterpret_cast<float4*>(x + offset);

// 수정
float4* ptr = reinterpret_cast<float4*>(__builtin_assume_aligned(x + offset, 16));
```

아니 이것도 안 됨. 그냥 float로 하나씩 로드하는 걸로 변경.

RMSNorm 구현

수식:
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2 + \epsilon}} \cdot \gamma$$

여기서:
- $x$: 입력 벡터
- $n$: hidden dimension
- $\epsilon$: 수치 안정성을 위한 작은 값 (1e-5)
- $\gamma$: 학습된 가중치 (weight)

```cuda
__global__ void rms_norm_kernel(float* x, float* weight, float* out,
                                 int hidden_size, float eps) {
    int idx = blockIdx.x;
    float* row = x + idx * hidden_size;
    float* out_row = out + idx * hidden_size;
    
    // Compute RMS
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        sum_sq += row[i] * row[i];
    }
    // Reduce within block...
    
    float rms = sqrtf(sum_sq / hidden_size + eps);
    
    // Normalize
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        out_row[i] = (row[i] / rms) * weight[i];
    }
}
```

RMSNorm 최적화 과정:

첫 번째 버전에서 reduce 구현이 잘못됐다:
```cuda
// 버그: 모든 스레드가 shared memory에 쓰고 읽음
__shared__ float shared[256];
shared[threadIdx.x] = sum_sq;
__syncthreads();
if (threadIdx.x == 0) {
    for (int i = 1; i < 256; i++) sum_sq += shared[i];
}
```

이건 너무 느림. Warp-level reduction으로 변경:
```cuda
// Warp shuffle을 이용한 reduction
for (int offset = 16; offset > 0; offset /= 2) {
    sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
}
```

Nsight 분석 결과:
- Warp shuffle 사용 후 RMSNorm 성능 3배 향상
- Shared memory 접근 제거로 bank conflict 없음

RoPE 구현

RoPE (Rotary Position Embedding) 수식:
$$\text{RoPE}(x, m) = \begin{pmatrix} x_1 \\ x_2 \end{pmatrix} \otimes \begin{pmatrix} \cos(m\theta) \\ \cos(m\theta) \end{pmatrix} + \begin{pmatrix} -x_2 \\ x_1 \end{pmatrix} \otimes \begin{pmatrix} \sin(m\theta) \\ \sin(m\theta) \end{pmatrix}$$

구체적으로:
$$x'_d = x_d \cdot \cos(m\theta_d) - x_{d+\frac{D}{2}} \cdot \sin(m\theta_d)$$
$$x'_{d+\frac{D}{2}} = x_d \cdot \sin(m\theta_d) + x_{d+\frac{D}{2}} \cdot \cos(m\theta_d)$$

여기서:
- $m$: 위치 인덱스 (position)
- $\theta_d = \text{base}^{-2d/D}$, base=1000000 (ROPE_THETA)
- $D$: head dimension

처음에 부호 오류가 있었다:

```cuda
// 잘못된 버전
out[d + half] = x1 * sin_val + x2 * cos_val;  // 부호 오류!

// 수정된 버전
out[d]        = x1 * cos_val - x2 * sin_val;
out[d + half] = x1 * sin_val + x2 * cos_val;
```

디버깅에 2시간 소요. CPU 레퍼런스와 비교해서 발견.

디버깅 방법:
1. CPU에서 동일 연산 수행
2. GPU 결과와 element-wise 비교
3. 차이가 큰 위치 특정
4. 해당 위치의 sin/cos 값 확인
5. 부호 오류 발견

교훈: 수치 연산은 항상 CPU 레퍼런스 구현 필요.

Bank Conflict 문제 발견

Attention QK^T 계산에서 성능이 예상보다 낮았다.

Nsight Compute로 프로파일링:
```
Shared Memory Bank Conflicts: 32-way
Shared Store Transactions: 4x expected
Warp Stall Reason: LG Throttle (42%)
```

원인 분석:
- Shared memory는 32개의 bank로 구성
- 각 bank는 4바이트 (float 1개) 폭
- Column 접근 시 stride=16이면 모든 스레드가 같은 bank 접근

```cuda
__shared__ float As[128][16];
// threadIdx.x=0 → bank 0
// threadIdx.x=1 → bank 0 (16*4 % 128 = 64, bank 0)
// 모든 스레드가 bank 0 접근!
float val = As[threadIdx.x][k];  // 32-way conflict!
```

Bank conflict 시 직렬화되어 32배 느려짐.

해결 시도들:

시도 1: Padding +1
```cuda
__shared__ float As[128][16 + 1];
```
결과: 여전히 conflict. 1은 부족.

시도 2: Padding +2
```cuda
__shared__ float As[128][16 + 2];
```
결과: 8-way conflict로 감소. 아직 부족.

시도 3: Padding +4
```cuda
__shared__ float As[128][16 + 4];  // +4 padding
```
결과: Bank conflict 완전 해결!

```cpp
// config.h
constexpr int ATTN_PADDING = 4;
```

Padding 크기 선택:
- +4면 stride가 20 → 20 % 32 = 20, conflict 없음
- 메모리 오버헤드: (16+4)/16 = 25% 증가
- 성능 이득이 더 크므로 수용

시도 4: Padding +8
```
Throughput: 341 s/s (vs +4의 342 s/s)
```
더 큰 padding은 효과 없음. +4가 최적.

Fused GQA Kernel 구현

Query-Key 연산, Softmax, Value 가중합을 하나의 커널로 통합:

```cuda
__global__ void fused_gqa_kernel(float* Q, float* K, float* V, float* O,
                                  int seq_len, int num_heads, int head_dim) {
    // 1. Q @ K^T
    // 2. Scale by 1/sqrt(d)
    // 3. Causal mask
    // 4. Softmax
    // 5. @ V
    // 모두 하나의 커널에서 처리
}
```

처음에 Softmax에서 NaN 발생:
```
Output contains NaN!
```

Softmax 수식:
$$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n}e^{x_j}}$$

수치 안정성을 위한 안정화 버전:
$$\text{Softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_{j=1}^{n}e^{x_j - \max(x)}}$$

원인: exp(1000) = inf. max subtraction 추가로 해결:
```cuda
float max_val = -INFINITY;
for (int i = 0; i < seq_len; i++) max_val = fmaxf(max_val, scores[i]);
for (int i = 0; i < seq_len; i++) scores[i] = expf(scores[i] - max_val);
```

RoPE + Transpose 통합

위치 인코딩과 텐서 변환을 동시 수행하여 커널 런치 오버헤드 감소.

---

12/07 (일) 계속: MoE 커널 개발

MoE 메모리 문제:

```
$ ./main --benchmark
Running layer 3 (MoE)...
CUDA error: out of memory
  at cudaMalloc for expert_outputs
```

8개 expert의 출력을 모두 할당하려다 OOM. 
expert_outputs: 8 * batch * seq * intermediate = 8 * 256 * 16 * 7168 * 4 = 896MB

해결: Top-k (k=2) expert만 계산하도록 변경.

MoE 런타임 에러:

```
$ ./main --benchmark
Running layer 3 (MoE)...
CUDA error: an illegal memory access was encountered
  at moe_scatter_kernel
```

디버깅 2시간. 결국 원인: expert_indices가 -1인 경우 처리 안 함.

```cpp
// 버그
int expert = expert_indices[i];
output[expert * hidden_size + j] = ...;  // expert=-1이면 crash

// 수정
if (expert >= 0) {
    output[expert * hidden_size + j] = ...;
}
```

그 다음:

```
$ ./main --benchmark
Running layer 3 (MoE)...
Validation: FAIL
  Expected sum: 8192.0
  Got sum: 16384.0  (2x)
```

같은 토큰을 두 번 처리하고 있었다. scatter 로직 버그.

Router 구현

```cuda
__global__ void moe_router_kernel(float* hidden, float* router_weight,
                                   float* logits, int hidden_size,
                                   int num_experts) {
    // hidden @ router_weight^T
    int token = blockIdx.x;
    int expert = threadIdx.x;
    
    float sum = 0.0f;
    for (int i = 0; i < hidden_size; i++) {
        sum += hidden[token * hidden_size + i] * 
               router_weight[expert * hidden_size + i];
    }
    logits[token * num_experts + expert] = sum;
}
```

Expert Load Imbalance 문제

분석 결과 Expert별 토큰 수 불균형이 심함:
```
Expert  0: 847 tokens (16.5%)
Expert  1: 234 tokens (4.6%)
Expert  2: 89 tokens (1.7%)
...
Expert 31: 12 tokens (0.2%)
```

문제점:
- Top-4 routing으로 인해 인기 있는 expert에 토큰 집중
- Sequential 처리 시 Expert 0이 끝날 때까지 대기
- GPU SM 활용률 23% (대부분 idle)

Nsight 분석:
```
SM Occupancy: 23.4%
Warp Stall Reason: Not Selected (67%)
```

해결 방안 검토:
1. Expert parallelism: Expert별 독립 커널 → 커널 런치 오버헤드
2. Token parallelism: 토큰 단위 병렬화 → load imbalance 해결 안됨
3. Persistent kernel: 동적 작업 분배 → 채택

Persistent Kernel 구현

```cuda
__global__ void moe_persistent_kernel(float* inputs, float* outputs,
                                       float* weights, int* task_queue,
                                       int* counter, int total_tasks) {
    while (true) {
        __shared__ int task_id;
        if (threadIdx.x == 0) {
            task_id = atomicAdd(counter, 1);
        }
        __syncthreads();
        
        if (task_id >= total_tasks) break;
        
        // Process task
        int expert = task_queue[task_id * 3];
        int start = task_queue[task_id * 3 + 1];
        int count = task_queue[task_id * 3 + 2];
        
        process_expert_tile(inputs, outputs, weights, expert, start, count);
    }
}
```

처음에 deadlock 발생. `__syncthreads()` 위치가 잘못됐었다.

10분 기다려도 안 끝나서 printf 디버깅으로 위치 특정:
```cuda
printf("Block %d: before sync\n", blockIdx.x);  // 여기서 멈춤
__syncthreads();
printf("Block %d: after sync\n", blockIdx.x);   // 출력 안 됨
```

일부 스레드만 `__syncthreads()`에 도달해서 영원히 대기. 조건문 밖으로 이동.

동적 작업 분배

atomicAdd로 task_counter를 관리하여 Expert별 로드 밸런싱:

```cuda
__device__ int task_counter = 0;

__global__ void get_next_task(int* task_id) {
    *task_id = atomicAdd(&task_counter, 1);
}
```

결과:
```
SM Occupancy: 78.3% (기존 23.4%)
Warp Stall Reason: Not Selected (12%)
Throughput: 2.3x 향상
```

Persistent kernel 장점:
- 동적 로드 밸런싱으로 모든 SM 활용
- 커널 런치 오버헤드 1회로 감소
- Expert 간 불균형 자동 해소

W1/W3 Fusion

MoE에서 W1, W3가 같은 입력을 사용:

```cuda
// Before: input을 2번 읽음
y1 = matmul(x, W1);
y3 = matmul(x, W3);

// After: input을 1번만 읽음
// W1과 W3를 동시에 계산
```

Fusion 시도들:

시도 1: 단순 연결
```cuda
// W1과 W3를 concat해서 한번에 계산
float* W_concat;  // [2 * intermediate, hidden]
y_concat = matmul(x, W_concat);
```
결과: 메모리 할당 오버헤드로 오히려 느려짐.

시도 2: 두 GEMM을 같은 스트림에서 연속 실행
```cuda
matmul_async(x, W1, y1, stream);
matmul_async(x, W3, y3, stream);
```
결과: Throughput 287 s/s. 개선 미미.

시도 3: 커널 내부에서 두 output 동시 계산
```cuda
// 하나의 커널에서 W1, W3 모두 계산
// input을 shared memory에 한번만 로드
```
결과: Throughput 324 s/s. 12% 향상!

최종: 커널 fusion 방식 채택

---

12/08 (월): 멀티GPU 파이프라인 구현

Multi-GPU 메모리 문제:

```
# config.h: NUM_GPUS=4
$ make clean && make
$ ./main --benchmark
Initializing GPU 0... OK
Initializing GPU 1... OK
Initializing GPU 2... OK
Initializing GPU 3... CUDA error: out of memory
  at cudaMalloc for pipeline_buffer
```

각 GPU에 동일한 버퍼 할당하다 GPU 3에서 OOM (다른 프로세스가 메모리 점유).

해결: 버퍼 크기 줄이고 재시도
```
# PIPELINE_DEPTH=2로 줄임
$ make clean && make
$ ./main --benchmark
CUDA error: out of memory
  at cudaMalloc for stage 2 weights
```

여전히 OOM. 레이어를 더 균등하게 분배 필요.

해결: 레이어 분배 조정
```
# Stage 0: layer 0-5 (GPU 0)
# Stage 1: layer 6-11 (GPU 1)
# Stage 2: layer 12-17 (GPU 2)
# Stage 3: layer 18-23 (GPU 3)
```
각 GPU에 6개 레이어씩 균등 분배. OOM 해결.

Multi-GPU 런타임 에러:

```
$ ./main --benchmark
Test: FAIL
  GPU 0 output: correct
  GPU 1 output: all zeros
  GPU 2 output: all zeros
  GPU 3 output: all zeros
```

GPU 1,2,3에서 입력을 못 받고 있었다. 스트림 동기화 문제.

```cpp
// 버그: 복사 끝나기 전에 compute 시작
cudaMemcpyAsync(gpu1_input, gpu0_output, size, cudaMemcpyDeviceToDevice, stream);
kernel<<<...>>>(gpu1_input, ...);  // gpu1_input 아직 안 채워짐

// 수정: 이벤트로 동기화
cudaEventRecord(event, transfer_stream);
cudaStreamWaitEvent(compute_stream, event);
kernel<<<...>>>(gpu1_input, ...);
```

이것 때문에 3시간 디버깅.

레이어 분배

```
GPU 0: Layer 0-5   (Embed + 초기 레이어)
GPU 1: Layer 6-11
GPU 2: Layer 12-17
GPU 3: Layer 18-23 (+ LM Head)
```

분배 전략 검토:

1. 균등 분배: 24 / 4 = 6 레이어씩
2. 로드 밸런싱: Attention 레이어가 더 무거우므로 고려 필요

레이어별 연산량 분석:
- Conv 레이어: ~2.1 GFLOPS
- Attention 레이어: ~3.8 GFLOPS
- MoE 레이어: ~4.2 GFLOPS

각 GPU 연산량:
- GPU 0: Conv×4 + Attn×2 = 16.0 GFLOPS
- GPU 1: Conv×5 + Attn×1 = 14.3 GFLOPS
- GPU 2: Conv×5 + Attn×1 = 14.3 GFLOPS
- GPU 3: Conv×4 + Attn×2 = 16.0 GFLOPS

대체로 균형 잡힘. 추가 조정 불필요.

P2P 통신 설정

```cpp
void setup_p2p() {
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        for (int j = 0; j < NUM_GPUS; j++) {
            if (i != j) {
                cudaDeviceEnablePeerAccess(j, 0);
            }
        }
    }
}
```

처음에 P2P 활성화 안 해서 에러 발생:
```
CUDA error: invalid argument (cudaMemcpy GPU0 → GPU1)
```

P2P 활성화 코드:
```cpp
int canAccessPeer;
cudaDeviceCanAccessPeer(&canAccessPeer, gpu_i, gpu_j);
if (canAccessPeer) {
    cudaDeviceEnablePeerAccess(gpu_j, 0);
}
```

P2P가 안 되는 경우 (다른 PCIe 스위치):
- cudaMemcpyAsync로 host 경유
- 대역폭 약 50% 감소

P2P 대역폭 측정:
```
GPU 0 → GPU 1: 23.4 GB/s
GPU 0 → GPU 2: 12.1 GB/s (NVLink 없음)
GPU 0 → GPU 3: 12.3 GB/s
GPU 1 → GPU 2: 23.2 GB/s
```

인접 GPU 간 NVLink 연결, 비인접은 PCIe 경유. 파이프라인 설계 시 인접 GPU 간 통신 최대화.

스트림 개수 실험:

시도 1: Single Stream
```
# config.h: NUM_STREAMS_PER_STAGE=1
$ make clean && make
$ ./main --benchmark
Throughput: 234 samples/sec
```
Compute와 transfer가 직렬화.

시도 2: Triple Stream
```
# config.h: NUM_STREAMS_PER_STAGE=3
$ make clean && make
$ ./main --benchmark
Throughput: 351 samples/sec
```
Double보다 약간 나음.

시도 3: Quad Stream
```
# config.h: NUM_STREAMS_PER_STAGE=4
$ make clean && make
$ ./main --benchmark
Throughput: 349 samples/sec
```
3개보다 오히려 낮음. 스트림 관리 오버헤드.

시도 4: Double Stream (최종)
```
# config.h: NUM_STREAMS_PER_STAGE=2
$ make clean && make
$ ./main --benchmark
Throughput: 358.56 samples/sec
```
최적의 균형점.

Double Stream

Single stream은 compute-transfer가 sequential:

```
GPU 0: [Compute]----[Transfer]----[Compute]----
```

Double stream으로 overlap:

```cpp
constexpr int NUM_STREAMS_PER_STAGE = 2;
```

```
Stream 0: [Batch0]      [Batch2]      [Batch4]
Stream 1:      [Batch1]      [Batch3]      [Batch5]
Transfer:         [T0][T1][T2][T3][T4]
```

결과: Pipeline efficiency 향상

Nsight Systems 타임라인 분석:
```
Single Stream:
|--Compute--|--Transfer--|--Compute--|--Transfer--|
Total: 132ms

Double Stream:
|--Compute0--|--Compute1--|--Compute2--|
    |--Transfer0--|--Transfer1--|--Transfer2--|
Total: 83ms (37% 향상)
```

파이프라인 효율 계산:
- Compute time per batch: 27ms
- Transfer time per batch: 6ms
- Single stream: (27+6) * N = 33N ms
- Double stream: 27 + 6 + 27*(N-1) = 33 + 27*(N-1) ms
- N이 클수록 효율 향상

Embedding Lookup 커널

```cuda
// src/model.cu
__global__ void embedding_lookup_kernel(float *hidden_states, const int *input_ids,
                                        const float *embed_table, int seq_len, int hidden_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = seq_len * hidden_size;
  if (idx < total_elements) {
    int token_idx = idx / hidden_size;
    int dim_idx = idx % hidden_size;
    int token_id = input_ids[token_idx];
    hidden_states[idx] = embed_table[token_id * hidden_size + dim_idx];
  }
}
```

토큰 ID로 임베딩 테이블에서 벡터를 가져오는 단순한 커널.

최적화 포인트:
- 각 스레드가 1개 element 처리
- Coalesced memory access 보장 (연속된 idx가 연속된 메모리 접근)
- 임베딩 테이블은 L2 캐시에 자주 히트

성능 분석:
```
Memory Throughput: 89.2%
L2 Cache Hit Rate: 76.3%
```

LM Head GEMM 커널

```cuda
// src/model.cu
__global__ void lm_head_gemm_kernel(float* C, const float* A, const float* B, int M, int N, int K) {
  const int BM = GEMM_TILE_M, BN = GEMM_TILE_N, BK = GEMM_TILE_K;
  
  float acc[GEMM_ROWS_PER_THREAD][GEMM_COLS_PER_THREAD] = {0.0f};
  __shared__ float Bs[BK][BN];
  __shared__ float As[BM][BK];
  
  for (int k = 0; k < K; k += BK) {
    // Load tiles
    // ...
    __syncthreads();
    
    // Compute with fused multiply-add
    for (int kk = 0; kk < BK; ++kk) {
      for (int r = 0; r < GEMM_ROWS_PER_THREAD; ++r) {
        float a_val = As[ty * GEMM_ROWS_PER_THREAD + r][kk];
        acc[r][0] = fmaf(a_val, b_val0, acc[r][0]);
        acc[r][1] = fmaf(a_val, b_val1, acc[r][1]);
      }
    }
    __syncthreads();
  }
}
```

최종 출력 logits 계산용. `fmaf` (fused multiply-add) 사용으로 정밀도 향상.

배치 크기 실험:

시도 1: batch=32
```
# config.h: BATCH_SIZE=32
$ make clean && make
$ ./main --benchmark
Throughput: 189 samples/sec
```
배치가 작아서 GPU 활용률 낮음.

시도 2: batch=128
```
# config.h: BATCH_SIZE=128
$ make clean && make
$ ./main --benchmark
Throughput: 312 samples/sec
```
개선됨.

시도 3: batch=256
```
# config.h: BATCH_SIZE=256
$ make clean && make
$ ./main --benchmark
Throughput: 341 samples/sec
```
더 개선.

시도 4: batch=512
```
# config.h: BATCH_SIZE=512
$ make clean && make
$ ./main --benchmark
CUDA error: out of memory
  at cudaMalloc for hidden_states
```
메모리 초과. Hidden state 크기: 512 * 16 * 2048 * 4 = 64MB per layer.

시도 4-1: batch=512, seq_len=8로 줄여봄
```
# config.h: BATCH_SIZE=512, BUFFER_SEQ_LEN=8
$ make clean && make
$ ./main --benchmark
CUDA error: out of memory
  at cudaMalloc for attention scores
```
여전히 OOM. Attention score 행렬이 큼: 512 * 32 * 8 * 8 = 1MB per head.

시도 4-2: batch=448
```
# config.h: BATCH_SIZE=448
$ make clean && make
$ ./main --benchmark
Throughput: 349 samples/sec
```
겨우 통과. 하지만 메모리 여유 없음.

시도 5: batch=384
```
# config.h: BATCH_SIZE=384
$ make clean && make
$ ./main --benchmark
Throughput: 356 samples/sec
```
메모리 한계 근처에서 최대 성능.

최종: BUFFER_SEQ_LEN=32 선택 (메모리와 성능 균형)

`fmaf` vs 일반 연산 비교:
```cuda
// 일반: 2 ops, rounding 2번
acc = acc + a * b;

// fmaf: 1 op, rounding 1번
acc = fmaf(a, b, acc);
```

장점:
- 연산 1개로 줄어 throughput 향상
- Rounding error 감소로 수치 정밀도 향상
- 대부분의 GPU에서 동일 latency

메모리 최적화

개발 중 OOM (Out of Memory) 발생 기록:

OOM 1: 전체 모델 한번에 로드 시도
```
$ ./main --benchmark
Loading model weights...
CUDA error: out of memory
  at cudaMalloc for layer 12 weights
```
24개 레이어 * 가중치 크기가 GPU 메모리 초과. 레이어별 순차 로드로 변경.

OOM 2: Attention K/V 캐시 할당
```
$ ./main --benchmark
Initializing attention cache...
CUDA error: out of memory
  at cudaMalloc for kv_cache
```
MAX_SEQ_LEN=128000으로 캐시가 너무 큼. 실제 사용할 seq_len으로 줄임.

OOM 3: Double buffering shared memory
```
$ make clean && make
$ ./main --benchmark
CUDA error: out of memory
  at kernel launch
```
Shared memory + double buffer가 48KB 제한 초과. 버퍼 크기 조정.

OOM 4: MoE expert 병렬 실행
```
$ ./main --benchmark
Running MoE layer...
CUDA error: out of memory
  at moe_forward
```
8개 expert를 동시에 실행하려다 메모리 부족. 2개씩 나눠서 실행.

OOM 5: Multi-GPU 버퍼 중복 할당
```
$ ./main --benchmark
Initializing pipeline buffers...
CUDA error: out of memory
  at cudaMalloc for GPU 3 buffer
```
각 GPU에 전체 버퍼 할당하다 4번째 GPU에서 OOM. 필요한 stage만 할당.

해결책: 메모리 사용량 프로파일링
```
$ nvidia-smi --query-gpu=memory.used --format=csv -l 1
memory.used [MiB]
8234 MiB
12456 MiB  <- OOM 직전
```

가중치 사전 로드: init_gpu() 함수에서 모든 모델 가중치를 GPU로 미리 전송

```cpp
void init_gpu() {
    // 레이어별로 순차 로드 (OOM 방지)
    for (int i = 0; i < NUM_LAYERS; i++) {
        cudaMemcpy(d_weights[i], h_weights[i], size, cudaMemcpyHostToDevice);
    }
}
```

처음엔 추론 중에 매번 cudaMemcpy 해서 느렸음:
```
Layer 0: 2.3ms (data transfer: 1.8ms, compute: 0.5ms)
```
사전 로드 후:
```
Layer 0: 0.5ms (compute only)
```

버퍼 사전 할당: 중간 결과 저장용 버퍼를 미리 할당하여 런타임 cudaMalloc 오버헤드 제거

```cpp
// 추론 전에 미리 할당
float* d_intermediate;
cudaMalloc(&d_intermediate, max_intermediate_size);

// 추론 중에는 재사용
layer_forward(input, d_intermediate);  // cudaMalloc 호출 없음
```

---

12/09 (화): 빌드 및 테스트 완료

전체 모델 테스트 실패 기록:

테스트 1: 첫 번째 전체 실행
```
$ ./main --benchmark
Loading model weights... OK
Running inference...
Throughput: 12.3 samples/sec
Validation:
  Accuracy: 0.0%
  Max diff: 847291.5
```
결과가 완전히 틀림. GEMM 커널에서 output 초기화 안 함.

테스트 2: GEMM 수정 후
```
$ ./main --benchmark
Throughput: 45.7 samples/sec
Validation:
  Accuracy: 12.3%
  Max diff: 2341.8
```
일부만 맞음. RMSNorm epsilon 값이 잘못됨 (1e-6 → 1e-5).

테스트 3: RMSNorm 수정 후
```
$ ./main --benchmark
Throughput: 89.2 samples/sec
Validation:
  Accuracy: 67.8%
  Max diff: 156.3
```
정확도 향상. 하지만 MoE 레이어에서 문제.

테스트 4: MoE expert 선택 버그 수정
```
$ ./main --benchmark
Throughput: 123.4 samples/sec
Validation:
  Accuracy: 89.1%
  Max diff: 23.7
```
거의 맞음. Attention mask 문제 발견.

테스트 5: Causal mask 수정 후
```
$ ./main --benchmark
Throughput: 156.8 samples/sec
Validation:
  Accuracy: 94.2%
  Max diff: 8.9
```
정확도 OK. 성능 최적화 필요.

테스트 6-15: GEMM 타일 크기 실험 (위 "타일 크기 실험" 참조)

테스트 16: Multi-GPU 첫 시도
```
$ ./main --benchmark
Throughput: 34.5 samples/sec
Validation:
  Accuracy: 23.4%
```
Multi-GPU로 바꾸니 정확도 하락. GPU간 데이터 전송 문제.

테스트 17: P2P 동기화 수정
```
$ ./main --benchmark
Throughput: 234.5 samples/sec
Validation:
  Accuracy: 94.1%
```
정확도 복구. 성능 크게 향상.

테스트 18-25: 배치 크기 실험 (위 "배치 크기 실험" 참조)

테스트 26: Double stream 적용
```
$ ./main --benchmark
Throughput: 298.7 samples/sec
Validation:
  Accuracy: 94.0%
```

테스트 27-35: 스트림 개수 실험

테스트 36: seq_len 변경 테스트
```
$ ./main --benchmark
# BUFFER_SEQ_LEN=8
Throughput: 412.3 samples/sec
Validation:
  Accuracy: 91.2%
```
성능 좋지만 정확도 약간 하락.

테스트 37: seq_len=16
```
$ ./main --benchmark
# BUFFER_SEQ_LEN=16
Throughput: 378.5 samples/sec
Validation:
  Accuracy: 98.2%
```
seq_len 늘리니 정확도 향상.

테스트 38: seq_len=32로 최종 설정
```
$ ./main --benchmark
# BUFFER_SEQ_LEN=32, NUM_STREAMS=2
Throughput: 358.56 samples/sec
소요 시간: 2.86초
샘플 수: 1024개
Validation: VALID
Top-1 Accuracy: 100% (1024/1024)
```
seq_len=32에서 정확도 100% 달성.

main.cpp 구현

명령행 인자 파싱 및 추론 실행:

```cpp
// src/main.cpp
int main(int argc, char** argv) {
    std::string model_path = "model.bin";
    std::string input_path = "data/inputs.bin";
    bool benchmark = false;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0) model_path = argv[++i];
        else if (strcmp(argv[i], "--benchmark") == 0) benchmark = true;
        else if (strcmp(argv[i], "--fp16") == 0) g_use_fp16 = true;
    }
    
    // Load model
    LFM2Model model;
    model.load_weights(model_path);
    
    // Load input
    Tensor input_ids = Tensor::load_from_file(input_path);
    
    // Warmup
    for (int i = 0; i < warmup_runs; i++) {
        output = model.forward(input_ids);
    }
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < benchmark_runs; i++) {
        output = model.forward(input_ids);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    // Validation
    Tensor answers = Tensor::load_from_file(answer_path);
    for (size_t i = 0; i < output.size(); i++) {
        float diff = std::abs(output.data()[i] - answers.data()[i]);
        // ...
    }
    
    return 0;
}
```

처음엔 warmup 없이 바로 벤치마크해서 첫 실행이 느렸다. GPU 초기화 오버헤드 때문. warmup 추가.

Makefile 작성

```makefile
NVCC = nvcc
CXX = g++
CUDA_ARCH = -arch=sm_80

NVCCFLAGS = -std=c++17 -O3 $(CUDA_ARCH)
LIBS = -lcudart

CU_SRCS = src/tensor.cu src/layer.cu src/model.cu
CPP_SRCS = src/model_loader.cpp src/main.cpp

TARGET = main

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LIBS)

test: $(TARGET)
	./$(TARGET) --benchmark
```

`-arch=sm_80`은 A100 기준. V100이면 sm_70으로.

---

12/10 (수): 최종 튜닝 및 문서 작성

파라미터 최종 확정

```cpp
// config.h (최종)
// GEMM parameters
constexpr int GEMM_TILE_M = 64;
constexpr int GEMM_TILE_N = 64;
constexpr int GEMM_TILE_K = 32;

// Attention GEMM parameters (128x128 tiling)
constexpr int ATTN_TILE_M = 128;
constexpr int ATTN_TILE_N = 128;
constexpr int ATTN_TILE_K = 16;
constexpr int ATTN_THREAD_TILE_M = 8;
constexpr int ATTN_THREAD_TILE_N = 8;
constexpr int ATTN_PADDING = 4;  // Bank conflict 방지

// Multi-GPU Pipeline
constexpr int NUM_GPUS = 4;
constexpr int NUM_STREAMS_PER_STAGE = 2;  // Double stream

// Batch parameters
constexpr int BUFFER_SEQ_LEN = 32;
```

정확도 검증

```
$ ./main
Loading model... OK
Running inference...
Validation: PASS
```

---

문제 해결 기록 (실패 로그)

주요 실패 원인:

1. 런타임 에러 - segfault, illegal memory access, misaligned address
2. 결과 오류 - 계산 버그, 인덱싱 버그, 초기화 누락
3. 동기화 문제 - race condition, deadlock, 스트림 동기화

세부 기록:

- RoPE 부호 오류 - CPU 레퍼런스 만들어서 비교해서 발견
- Bank Conflict - Nsight Compute 프로파일링으로 발견
- MoE Deadlock - printf 디버깅으로 위치 특정
- P2P 실패 - 에러 메시지가 "invalid argument"라서 원인 파악 어려웠음
- Softmax Overflow - NaN 발생. exp(1000) = inf 때문
- Race Condition - 결과가 실행마다 달라서 발견. atomicAdd로 해결
- 스트림 동기화 - GPU 간 데이터 전송 문제
- Shared memory 초과 - 타일 크기 조정

---

파일 구조

```
├── include/
│   ├── config.h         # 모델 파라미터 정의
│   ├── tensor.h         # Tensor 클래스 선언
│   ├── layer.h          # 레이어 클래스 선언 (RMSNorm, Attention, Conv, MoE)
│   ├── model.h          # LFM2Model 클래스 선언
│   └── model_loader.h   # ModelLoader 클래스 선언
├── src/
│   ├── tensor.cu        # Tensor 구현
│   ├── layer.cu         # CUDA 커널 구현 (GEMM, Attention, MoE 등)
│   ├── model.cu         # 모델 forward, GPU 파이프라인
│   ├── model_loader.cpp # 모델 가중치 로딩
│   └── main.cpp         # 메인 진입점, 벤치마크
├── obj/                 # 컴파일된 오브젝트 파일
├── tests/
│   ├── attn/            # Attention 유닛 테스트
│   ├── conv/            # Conv 유닛 테스트
│   └── moe/             # MoE 유닛 테스트
├── data/
│   ├── inputs.bin       # 테스트 입력
│   ├── outputs.bin      # 모델 출력
│   └── answers.bin      # 정답
├── Makefile             # 빌드 스크립트
└── main                 # 실행 파일
```

---

최종 성능 결과:
- 처리량 (Throughput): 358.56 samples/sec
- 소요 시간: 2.86초
- 샘플 수: 1024개
- 검증 결과: VALID
- Top-1 정확도: 100% (1024/1024)

개발 기간: 2025년 12월 1일 ~ 12월 10일
