from attention_mask import *


def run(_model, _optimizer, _criterion, _split):
    # 학습이냐 평가냐, 모드
    _model.train() if _split == "train" else _model.eval()

    _data_iter = Multi30k(split=_split, language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    _dataloader = DataLoader(_data_iter, batch_size=BATCH_SIZE, collate_fn=collator)

    losses = 0

    for source_batch, target_batch in _dataloader:
        source_batch = source_batch.to(DEVICE)
        target_batch = target_batch.to(DEVICE)

        _target_input = target_batch[:-1, :]
        _target_output = target_batch[1:, :]

        _src_mask, _tgt_mask, _src_padding_mask, _tgt_padding_mask = create_mask(
            source_batch, _target_input
        )


        logits = _model(src=source_batch,
                        trg=_target_input,
                        src_mask=_src_mask,
                        tgt_mask=_tgt_mask,
                        src_padding_mask=_src_padding_mask,
                        tgt_padding_mask=_tgt_padding_mask,
                        memory_key_padding_mask=_src_padding_mask)

        _optimizer.zero_grad()
        loss = _criterion(logits.reshape(-1, logits.shape[-1]), _target_output.reshape(-1))

        if _split == "train":
            loss.backward()
            _optimizer.step()

        loss += loss.item()

    return losses / len(list(_dataloader))


for epoch in range(5):
    train_loss = run(model, optimizer, criterion, "train")
    val_loss = run(model, optimizer, criterion, "valid")
    print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.3f}, Val loss: {val_loss:.3f}")

torch.save(model.state_dict(), f"./models/de2en_transformer_{DEVICE}.pt")

# 6