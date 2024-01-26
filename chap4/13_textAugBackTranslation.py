import nlpaug.augmenter.word as naw

def foo():
    texts = [
        "Those who can imagine anything, can create the impossible",
        "We can only see a short distance ahead, but we can see plenty there that needs to be done.",
        "If a machine is expected to be infallible, it cannot also be intelligent."
    ]

    back_translation = naw.BackTranslationAug(from_model_name='facebook/wmt19-en-de',
                                              to_model_name='facebook/wmt19-de-en')
    augmented_texts = back_translation.augment(texts)

    for text, augmented in zip(texts, augmented_texts):
        print(f"src : {text}")
        print(f"dst : {augmented}")
        print("----------------")

if __name__=='__main__':
    foo()

# 걍 하면 안되고 if __name__이 필요하다
# freeze_support() error의 해결법이라고 한다
# window 환경에서 *fork미지원으로 인해 부모자식프로세스간에 구분이 되지않아 프로세스를 계속 불러오는 오류가 생기는 모양...
# spawn은 복제 생성이 아니라 새로운 프로세스 시작인 모양