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

flashattention technique
https://www.youtube.com/watch?v=gMOAud7hZg4&t=520s
1.tiling

2.recomputation
forward pass에서 구한 attention matrix를 저장하지 않고 backward pass때 바로 recompute 진행

3. softmax 전에 max 값 빼주기
모든 값들이 수치적으로 안정화 : 큰 숫자 계산 x

![image](https://github.com/jinuk0211/flashattention/assets/150532431/f0710580-8b4f-4142-9d91-65e6eee63f86)


flashattention2

기존 softmax
![image](https://github.com/jinuk0211/flashattention/assets/150532431/775b4abc-001a-4fae-ad8b-d24929a0aece)

local softmax
![image](https://github.com/jinuk0211/flashattention/assets/150532431/523c0ed9-40f9-4104-be5f-07dc16405dcd)


매트릭스 곱셈이 아닌 non matmul FLOP 감소

312 TFLOPs/s of
FP16/BF16 matmul 
19.5 TFLOPs/s of
non-matmul FP32.


thread : CUDA 에서의 개별 작업 단위
warp : nvidia gpu에서 동시에 실행되는 thread 그룹, 일반적으로 32개, 이 그룹이 동일한 명령어를 실행하는 것, 프로그래머는 개별 스레드보다 warp 단위로 작업을 최적화하는 것이 좋음

https://tridao.me/publications/flash2/flash2.pdf

![image](https://github.com/jinuk0211/flashattention/assets/150532431/408302a0-7e86-4096-908a-52852219bdae)
![image](https://github.com/jinuk0211/flashattention/assets/150532431/90d42373-5d13-4bd9-8bcd-4e759d9643c2)
