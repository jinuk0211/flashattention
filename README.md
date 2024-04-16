flashattention

flashattention v2

infiniattention

하드웨어적 기술 - gpu 메모리 사용효율 극대화 

v1 : 속도 2-4배 ↑ , 메모리 사용량 10-20배 감소
v2 : 속도 4-8배, 메모리 사용량 20-40배 감소


엔비디아 A100 gpu 메모리계층 ↓


현대 gpu 특징

https://tridao.me/publications/flash2/flash2.pdf
1.메모리에 접근하는 능력(memory access) <<< 컴퓨팅 능력(matmul e.g  Tensor Cores on Nvidia GPUs)
2. nonmatmul 연산이 matmul 연산보다 16배 구림(expensive) - A100 gpu
3. 기본적으로 HBM에 저장하고 SRAM에서 계산 수행


이 과정 최적화(softmax trick, recomputation 등) -> flashattention 


![image](https://github.com/jinuk0211/flashattention/assets/150532431/2f4c5427-46ec-4d8f-a2b5-64b92bf774d8)



flashattetion 이점
1. 속도 관련
2. 메모리 관련
3. long context

본질적으로 트랜스포머가 long context에 적합하지 않음에도 이 관련 문제를 해결하려는 이유 :

1. 모델 성능 향상에 도움이 될 것 (promising)
2. high-resolution 이미지 이해에 도움이 될 것
3. 비디오 생성, 오디오, DNA sequence, code와 같은 분야에 트랜스포머 적용이 수월해짐
 
그냥 데이터를 crop하고 compressing해서 학습시키면 안되나?
해도 됨 -> https://arxiv.org/abs/2404.03626?utm_source=pytorchkr
Equal-Info Windows 사용한 논문 

flashattention

요점
1. 전체 sequence 즉 full input에 접근하지 않고 softmax를 수행
2. forward pass로 부터의 어텐션 matrix 저장없이 backward 수행


flashattention technique
https://www.youtube.com/watch?v=gMOAud7hZg4&t=520s
1.tiling
 Q,K,V matrices를 block단위로 쪼갠 뒤 softmax를 적용하고, 다시 합칠 수 있다는 것 -> 기존의 큰 softmax가 블럭으로 쪼개지며 수행되기 때문에 recomputation에도 도움이 됨

2. recomputation - kernel fusion, mamba에도 사용
forward pass에서 구한 attention matrix를 저장하지 않고 backward pass때 바로 recompute 진행, intermediate(중간 산물) N x N matrix를 HBM에 읽고 쓰는 과정이 없어짐

KV cache와 비교
KV cache는 key와 value자체의 메모리 접근량은 줄어도 kv cache에 특정 key, value 위치를 특정(cache에 저장)하기 위해 오히려 메모리 사용량 자체가 늘어남, 동적으로 크기가 변동되니 다루기 어려움
속도<-> 메모리 간의 tradeoff 발생, (애초에 하드웨어적 최적화가 아님, 추론 속도를 극대화하기 위함) - long context 경우는 더 악영향

3. softmax 전에 max 값 빼주기
모든 값들이 수치적으로 안정화 : 큰 숫자 계산 x

기존 softmax
->
flash attention으로의 local softmax - 
지수 분리, diag()는 causal masking의미
: normalization
: full -> local

![image](https://github.com/jinuk0211/flashattention/assets/150532431/f0710580-8b4f-4142-9d91-65e6eee63f86)


flashattention2

기존 softmax
![image](https://github.com/jinuk0211/flashattention/assets/150532431/775b4abc-001a-4fae-ad8b-d24929a0aece)

local softmax
![image](https://github.com/jinuk0211/flashattention/assets/150532431/523c0ed9-40f9-4104-be5f-07dc16405dcd)


Parallelism, and Work Partitioning, 더 나은 online softmax trick 

thread : CUDA 에서의 개별 작업 단위
warp : nvidia gpu에서 동시에 실행되는 thread 그룹, 일반적으로 32개 thread, 이 그룹이 동일한 명령어를 실행하는 것, 프로그래머는 개별 스레드보다 warp 단위로 작업을 최적화하는 것이 좋음

1.parallism

flashattention 에서는 배치사이즈와 heads 개수로 병렬화, 각각의 thread 블럭이 하나의 attention head를 처리함.(nums>80에서 효과적) gpu 리소스를 효율적으로 사용 가능했음
long context에는 작은 heads 숫자와 배치 사이즈 자체가 작기 때문에(seq len이 길어서 어쩔 수 없음 - 한번에 큰 데이터를 처리할 기술여건 x) -> seq len dim에 병렬화를 추가로 수행 -> 속도 증가

https://tridao.me/publications/flash2/flash2.pdf

2. work partitioning
한 thread 블럭에 여러개의 warp로 분산해서 처리함 
워프 간의 통신과 동기화를 줄이기 위함 

flashattention - 4개의 워프로 key, value 분할하고 query 에 접근
단점 : 모든 워프가 중간 결과를 공유 메모리에 쓰고 동기화해야함

flashattention 2 -  K와 V를 모든 워프에 접근할 수 있도록 유지하면서 Q를 4개의 워프에 걸쳐 분할
보완점 : key x query matmul한 다음 value 행렬 곱하면 됨, 워프간의 통신 필요 없어짐 



[매트릭스 곱셈이 아닌 non matmul FLOP 감소

312 TFLOPs/s of
FP16/BF16 matmul 
19.5 TFLOPs/s of
non-matmul FP32.


thread : CUDA 에서의 개별 작업 단위
warp : nvidia gpu에서 동시에 실행되는 thread 그룹, 일반적으로 32개, 이 그룹이 동일한 명령어를 실행하는 것, 프로그래머는 개별 스레드보다 warp 단위로 작업을 최적화하는 것이 좋음

https://tridao.me/publications/flash2/flash2.pdf
](https://tridao.me/publications/flash2/flash2.pdf

2. work partitioning
한 thread 블럭에 여러개의 warp로 분산해서 처리함 
워프 간의 통신과 동기화를 줄이기 위함 

flashattention - 4개의 워프로 key, value 분할하고 query 에 접근
단점 : 모든 워프가 중간 결과를 공유 메모리에 쓰고 동기화해야함

flashattention 2 -  K와 V를 모든 워프에 접근할 수 있도록 유지하면서 Q를 4개의 워프에 걸쳐 분할
보완점 : key x query matmul한 다음 value 행렬 곱하면 됨, 워프간의 통신 필요 없어짐 

)
![image](https://github.com/jinuk0211/flashattention/assets/150532431/408302a0-7e86-4096-908a-52852219bdae)
![image](https://github.com/jinuk0211/flashattention/assets/150532431/90d42373-5d13-4bd9-8bcd-4e759d9643c2)


3. 더 나은 online softmax 
매트릭스 곱셈이 아닌 non matmul FLOP 을 줄이기

컴퓨팅 속도 차이 A100

312 TFLOPs/s of
FP16/BF16 matmul 
19.5 TFLOPs/s of
non-matmul FP32.

matmul
compute-bound - 컴퓨팅에 연산시간 대부분 소요
예시) matmul, convolution with large # of channels
non matmul
memory-bound - 메모리 접근에 연산시간 대부분 소요
예시) elementwise ops(activation, dropout), reduction(sum, softmax, batchnorm, layernorm)
1. diag 연산을 최소화 -> causal masking 연산 줄임 

2. softmax의 max m과 지수들의 합(summation)을 backward를 위해 저장하지 않음
