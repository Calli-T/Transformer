import nlpaug.augmenter.char as nac

texts = [
    "Those who can imagine anything, can create the impossible",
    "We can only see a short distance ahead, but we can see plenty there that needs to be done.",
    "If a machine is expected to be infallible, it cannot also be intelligent."
]

aug = nac.RandomCharAug(action='delete')
augmented_texts = aug.augment(texts)

for text, augmented in zip(texts, augmented_texts):
    print(f"src : {text}")
    print(f"dst : {augmented}")
    print("----------------")