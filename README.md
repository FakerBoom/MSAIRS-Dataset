# MSAIRS-Dataset

We are excited to announce that our paper *"Impact of Stickers on Multimodal Sentiment and Intent in Social Media: A New Task, Dataset and Baseline"* has been accepted for **ACM MM 2025**. The paper's new version will be available soon, and the link will be shared shortly. You can find the previous version of the paper on **arXiv** [here](https://arxiv.org/abs/2405.08427).

This repository contains the **MSAIRS dataset** and associated **code** for multimodal sentiment analysis and intent recognition involving stickers in social media conversations. The dataset and model implementation are designed to tackle the task of jointly predicting sentiment and intent in multimodal contexts, such as text and stickers. This resource is meant to assist researchers and practitioners in understanding and analyzing the influence of stickers in communication.

### Download Stickers

We have shared all the stickers from MSAIRS via Quark Cloud. Download them using the link below:

- **Sticker download link**: (https://pan.quark.cn/s/4f51fcfbfe51?pwd=nkUi)
- **Extraction code**: nkUi

Please copy the link or click on it, and open it with the Quark APP to get all the stickers.


### Dataset Example

The dataset contains the following key fields:

```json
{
    "id": 0,
    "context": "相信自己，明天会更好\t",
    "sticker": "123",
    "multimodal_intent_label": "Comfort",
    "multimodal_sentiment_label": 1,
    "text_sentiment_label": 1,
    "sticker_sentiment_label": 1,
    "sticker_class": 5,
    "sticker_text": "加油"
}
```
context: The text in the chat.

sticker: The ID of the sticker used, corresponding to an image file in the all_sticker folder.

multimodal_intent_label: The combined intent label derived from both text and sticker (e.g., Comfort).

multimodal_sentiment_label: The combined sentiment label (0 = Neutral, 1 = Positive, 2 = Negative).

text_sentiment_label: The sentiment label for the text alone (0 = Neutral, 1 = Positive, 2 = Negative).

sticker_sentiment_label: The sentiment label for the sticker alone (0 = Neutral, 1 = Positive, 2 = Negative).

sticker_class: The category of the sticker, where:

0: Person

1: Animal

2: Cartoon

3: Person with Text

4: Animal with Text

5: Cartoon with Text

6: Text Only

sticker_text: The text that appears on the sticker, if applicable.

Label Descriptions
Sentiment Labels:

0: Neutral

1: Positive

2: Negative

Intent Labels:

0: Complain

1: Praise

2: Agree

3: Compromise

4: Query

5: Joke

6: Oppose

7: Inform

8: Ask for help

9: Greet

10: Taunt

11: Introduce

12: Guess

13: Leave

14: Advise

15: Flaunt

16: Criticize

17: Thank

18: Comfort

19: Apologize

This dataset and associated code are open-source and intended to foster research and collaboration in the growing area of multimodal communication analysis.
