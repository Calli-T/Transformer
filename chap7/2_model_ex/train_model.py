from create_masks import *


def run(_model, _optimizer, _criterion, split):
    _model.train() if split == "train" else _model.eval()
    data_iter = Multi30k(split=split, language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    _dataloader = DataLoader(data_iter, batch_size=BATCH_SIZE, collate_fn=collator)

    losses = 0

    #count = 0
    for source_batch, target_batch in _dataloader:
        source_batch = source_batch.to(DEVICE)
        target_batch = target_batch.to(DEVICE)

        target_input = target_batch[:-1, :]
        target_output = target_batch[1:, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            source_batch, target_input
        )

        logits = _model(
            src=source_batch,
            trg=target_input,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_padding_mask=src_padding_mask,
            tgt_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )

        _optimizer.zero_grad()
        loss = _criterion(logits.reshape(-1, logits.shape[-1]), target_output.reshape(-1))
        if split == "train":
            loss.backward()
            _optimizer.step()
        losses += loss.item()
        # count += 1
        # print(count*128)

    return losses / len(list(_dataloader))


for epoch in range(5):
    train_loss = run(model, optimizer, criterion, "train")
    val_loss = run(model, optimizer, criterion, "valid")
    print(f"Epoch: {epoch + 1}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}")
    
# 도대체 토치(에서 쓰는 피클)은 왜 또 버그를 발생시키는가? 빈파일이 생성된다
# torch.save(model, f"./models/de2en_transformer_{DEVICE}.pt")

# 5