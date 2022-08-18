# papers-with-code

### TL;DR

```
PyTorch 기초부터 NLP 근본 딥러닝 모델의 논문 구현 스터디를 진행하고자 합니다.
```

### 스터디 룰

- 스터디 시간: 매주 수요일 저녁 8-10시, 첫 주만 13일(수) 오후 2-4시쯤 or 11일(월) 밤 10시
- 스터디 분량: 매주 논문 1개씩! (필요하다면 Attention 부터는 시간을 더 사용)
    - 매주 깃헙 이슈에 질문 하나씩 남기기!
- 발표자: 매주 2명씩 (2주에 한 번씩 발표할 수 있도록)
    - 발표 전날까지 깃허브에 본인 코드 업로드하기! (다른 팀원이 미리 코드를 보고 올 수 있도록)

### 구현 방향

- pytorch를 사용해 scratch부터 구현하는 것을 목표로!
- 모델 구현 후 loss 감소하는 것 확인 후, 자원이 가능하다면 논문에서 제시한 성능 재현하기 ([구글 클라우드 TPU 신청 링크](https://sites.research.google/trc/about/))

### 발표에 들어갈 내용

- 코드로 구현한 부분이 논문의 어떤 내용/수식에 해당하는지
- 논문 내용과 구현체에서 달라진 부분

### 참여자
> 이인서, 임수정, 한나연, 허치영, 김소연



| Date | Paper | Year | Presenter | Source |
|-------|-------|-------|-------|-------|
| 7/13 | [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385?context=cs) | 2015 | 인서, 나연 | [ResNet](https://github.com/HanNayeoniee/papers-with-code/tree/main/0713_cnn) |
| 7/20 | RNN, LSTM |  | 수정, 치영 | [RNN](https://github.com/HanNayeoniee/papers-with-code/tree/main/0721_rnn) |
| 7/27 | [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) | 2014 | 인서, 나연 | [Seq2Seq](https://github.com/HanNayeoniee/papers-with-code/tree/main/0727_seq2seq) |
| 8/3 | [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473?context=stat) | ICLR 2015 | 수정, 치영 | [Seq2Seq with Attention](https://github.com/HanNayeoniee/papers-with-code/tree/main/0803_attention) |
| 8/10 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762?context=cs) | NIPS 2017 | 모두 | [Transformer 리뷰]() |
| 8/17 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762?context=cs) | NIPS 2017 | 인서, 나연 | [Transformer 코드]() |
| 8/24 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762?context=cs) | NIPS 2017 | 수정, 소연 | [Transformer 코드]() |
| 8/31 | [Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) | 2018 | 예정 | [BERT 리뷰]() |
| 9/9 | [Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) | 2018 | 예정 | [BERT 코드]() |
