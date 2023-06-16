import re
import textwrap


def preprocess(text: str) -> str:
    text = text.replace("\n", "")
    text = re.sub("\s+", " ", text)
    return text


def process_response(response, mode) -> dict:
    ans = {}
    if mode == "Chat":
        ans["response"] = response["result"]
        ans["source_documents"] = response["source_documents"]
    if mode == "Search":
        ans["source_documents"] = response
    if mode == "Summary":
        wrapped_text = textwrap.fill(
            response["output_text"],
            width=100,
            break_long_words=False,
            replace_whitespace=False,
        )
        ans["summary"] = wrapped_text
    if mode == "OCR":
        ans["OCR"] = response
    if mode == "Conversation":
        ans["response"] = response
    return ans


def preprocess_ocr(docs):
    text = ""
    for block in docs["pages"][0]["blocks"]:
        for line in block["lines"]:
            for word in line["words"]:
                text += word["value"] + " "
    return text
