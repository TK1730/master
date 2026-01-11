import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate
import spacy
import ginza
from whisper_dataset import create_dataset


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self,
        features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # 音声とラベルの長さが異なるため、異なる長さでパディングするメソッドが必要
        # まずは音声入力の方から処理する PyTorchのTensorを返すようにする
        input_features = [
            {"input_features": feature["input_features"]}
            for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        # トークン化されたラベルをパディングする
        label_features = [
            {"input_ids": feature["labels"]}
            for feature in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # 損失の計算時に無視されるように、パディング部分を-100に置き換える
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # もし前のトークン化のステップでBOSトークン(文頭トークン)がすでに追加されている場合
        # ここで削除する
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        # # decoder_input_idsを作成 (labelsを右に1つシフト)
        # # -100をpad_token_idに置換してからシフトする
        # pad_token_id = self.processor.tokenizer.pad_token_id

        # # 1. -100 を pad_token_id に戻す
        # _labels = labels.clone()
        # _labels[_labels == -100] = pad_token_id

        # # 2. シフト (先頭に decoder_start_token_id を追加)
        # decoder_input_ids = _labels.new_zeros(_labels.shape)
        # decoder_input_ids[:, 1:] = _labels[:, :-1].clone()
        # decoder_input_ids[:, 0] = self.decoder_start_token_id

        # batch["decoder_input_ids"] = decoder_input_ids

        # # Debug: Check keys in batch
        # # print("Batch keys:", list(batch.keys()))
        # if "input_ids" in batch:
        #     batch.pop("input_ids")

        return batch


def compute_metrics(pred, metric, tokenizer, nlp):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # -100をパッドトークンidに変換
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # 評価指標(WER)を計算する際、トークンを(IDの塊として)グループ化するような処理は行わない
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # 分かち書きして空白切りに変換
    pred_str = [" ".join(str(i) for i in nlp(j)) for j in pred_str]
    label_str = [" ".join(str(i) for i in nlp(j)) for j in label_str]

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


# --- Main Training Function ---
def main():
    # datasetの作成
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small",
        language="Japanese",
        task="transcribe"
    )
    train_dataset, test_dataset = create_dataset(
        train_csv="dataset/train.csv",
        test_csv="dataset/test.csv",
        processor=processor
    )

    # modelの作成
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small"
    )
    # modelの言語とタスクを設定
    model.generation_config.language = "Japanese"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = (
        processor.get_decoder_prompt_ids(
            language="Japanese", task="transcribe"
        )
    )
    model.config.suppress_tokens = []
    # dropoutを設定
    model.config.dropout = 0.1
    model.config.attention_dropout = 0.1

    # spec augmentを有効化
    model.config.apply_spec_augment = True
    model.config.mask_time_prob = 0.05  # 5%の時間領域をマスク
    model.config.mask_feature_prob = 0.05  # 5%の頻域領域をマスク

    # # 転移学習: 最後の層(proj_out)以外を固定(Freeze)する
    # for param in model.parameters():
    #     param.requires_grad = False

    # # proj_out層のみ学習可能にする
    # if hasattr(model, "proj_out"):
    #     for param in model.proj_out.parameters():
    #         param.requires_grad = True
    # else:
    #     print("Warning: proj_out layer not found. Continuing with"
    #           "all parameters frozen or as is.")

    # # 学習可能なパラメータ数を確認
    # trainable_params = sum(
    #     p.numel() for p in model.parameters()
    #     if p.requires_grad
    # )
    # all_params = sum(p.numel() for p in model.parameters())
    # print(
    #     f"trainable params: {trainable_params} || "
    #     f"all params: {all_params} || "
    #     f"trainable%: {100 * trainable_params / all_params:.2f}"
    # )

    # # LoRAの設定
    # peft_config = LoraConfig(
    #     # task_type=TaskType.SEQ_2_SEQ_LM,
    #     inference_mode=False,
    #     r=32,
    #     lora_alpha=64,
    #     lora_dropout=0.1,
    #     target_modules=["q_proj", "v_proj"]
    # )
    # model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()

    # data collatorの作成
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id
    )

    # 推論方法の設定　WER(Words Error Rate)
    metric = evaluate.load("wer")
    nlp = spacy.load("ja_ginza")
    ginza.set_split_mode(nlp, "C")  # cはNEologの意味らしい

    # 学習要素の定義
    training_args = Seq2SeqTrainingArguments(
        output_dir="./output/whisper-small-ja-wer",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=10000,
        weight_decay=0.01,
        gradient_checkpointing=False,
        fp16=True,
        eval_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=448,
        save_steps=100,
        eval_steps=100,
        logging_steps=25,
        logging_dir="./output/whisper-small-ja",
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=1,
        push_to_hub=False,
    )

    # Gradient Checkpointingを使う場合はuse_cache=Falseにする必要がある
    # ここでは無効化しているが、念のため設定しておく
    model.config.use_cache = False

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=lambda preds: compute_metrics(
            preds, metric, processor.tokenizer, nlp
        ),
        tokenizer=processor.feature_extractor,
    )

    trainer.train()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
