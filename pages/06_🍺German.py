import pandas as pd
import re
import requests
import spacy
from spacy_streamlit import visualize_ner, visualize_tokens
#from spacy.language import Language
from spacy.tokens import Doc
import spacy_ke
import streamlit as st

# Global variables
DEFAULT_TEXT = """Im Schatten des Hauses, in der Sonne des Flußufers bei den Booten, im Schatten des Salwaldes, im Schatten des Feigenbaumes wuchs Siddhartha auf, der schöne Sohn des Brahmanen, der junge Falke, zusammen mit Govinda, seinem Freunde, dem Brahmanensohn. Sonne bräunte seine lichten Schultern am Flußufer, beim Bade, bei den heiligen Waschungen, bei den heiligen Opfern. Schatten floß in seine schwarzen Augen im Mangohain, bei den Knabenspielen, beim Gesang der Mutter, bei den heiligen Opfern, bei den Lehren seines Vaters, des Gelehrten, beim Gespräch der Weisen. Lange schon nahm Siddhartha am Gespräch der Weisen teil, übte sich mit Govinda im Redekampf, übte sich mit Govinda in der Kunst der Betrachtung, im Dienst der Versenkung. Schon verstand er, lautlos das Om zu sprechen, das Wort der Worte, es lautlos in sich hinein zu sprechen mit dem Einhauch, es lautlos aus sich heraus zu sprechen mit dem Aushauch, mit gesammelter Seele, die Stirn umgeben vom Glanz des klardenkenden Geistes. Schon verstand er, im Innern seines Wesens Atman zu wissen, unzerstörbar, eins mit dem Weltall.
"""
DESCRIPTION = "AI模型輔助語言學習：德語"
TOK_SEP = " | "
MODEL_NAME = "de_core_news_sm"
API_LOOKUP = {}
MAX_SYM_NUM = 5


# Utility functions

def create_de_df(tokens):
    seen_texts = []
    filtered_tokens = []
    for tok in tokens:
        if tok.lemma_ not in seen_texts:
            filtered_tokens.append(tok)

    df = pd.DataFrame(
        {
            "單詞": [tok.text.lower() for tok in filtered_tokens],
            "詞類": [tok.pos_ for tok in filtered_tokens],
            "原形": [tok.lemma_ for tok in filtered_tokens],
        }
    )
    st.dataframe(df)
    csv = df.to_csv().encode('utf-8')
    st.download_button(
        label="下載表格",
        data=csv,
        file_name='de_forms.csv',
    )

def filter_tokens(doc):
    clean_tokens = [tok for tok in doc if tok.pos_ not in ["PUNCT", "SYM"]]
    clean_tokens = [tok for tok in clean_tokens if not tok.like_email]
    clean_tokens = [tok for tok in clean_tokens if not tok.like_url]
    clean_tokens = [tok for tok in clean_tokens if not tok.like_num]
    clean_tokens = [tok for tok in clean_tokens if not tok.is_punct]
    clean_tokens = [tok for tok in clean_tokens if not tok.is_space]
    return clean_tokens


def create_kw_section(doc):
    st.markdown("## 關鍵詞分析")
    kw_num = st.slider("請選擇關鍵詞數量", 1, 10, 3)
    kws2scores = {keyword: score for keyword,
                  score in doc._.extract_keywords(n=kw_num)}
    kws2scores = sorted(kws2scores.items(), key=lambda x: x[1], reverse=True)
    count = 1
    for keyword, score in kws2scores:
        rounded_score = round(score, 3)
        st.write(f"{count} >>> {keyword} ({rounded_score})")
        count += 1


# Page setting
st.set_page_config(
    page_icon="🤠",
    layout="wide",
    initial_sidebar_state="auto",
)
st.markdown(f"# {DESCRIPTION}")

# Load the language model
nlp = spacy.load(MODEL_NAME)

# Add pipelines to spaCy
nlp.add_pipe("yake")  # keyword extraction
# nlp.add_pipe("merge_entities") # Merge entity spans to tokens

# Page starts from here
st.markdown("## 待分析文本")
st.info("請在下面的文字框輸入文本並按下Ctrl + Enter以更新分析結果")
text = st.text_area("",  DEFAULT_TEXT, height=200)
doc = nlp(text)
st.markdown("---")

st.info("請勾選以下至少一項功能")
keywords_extraction = st.checkbox("關鍵詞分析", False)
gender_analyzer = st.checkbox("詞性分析", True)
defs_examples = st.checkbox("單詞解析", True)
morphology = st.checkbox("詞形變化", False)
ner_viz = st.checkbox("命名實體", True)
tok_table = st.checkbox("斷詞特徵", False)

if keywords_extraction:
    create_kw_section(doc)

if gender_analyzer:
    st.markdown("## 分析後文本 (詞性)")
    for idx, sent in enumerate(doc.sents):
        enriched_sentence = []
        for tok in sent:
            if tok.pos_ == "NOUN":
                if not tok.morph.get("Gender"):
                    enriched_sentence.append(tok.text)
                else:
                    gender = tok.morph.get("Gender")[0]
                    print(dir(tok.morph))
                    if gender:
                        enriched_tok = f"{tok.text} ({gender})"
                        enriched_sentence.append(enriched_tok)
            else:
                enriched_sentence.append(tok.text)

        display_text = " ".join(enriched_sentence)
        st.write(f"{idx+1} >>> {display_text}")

if defs_examples:
    st.markdown("## 單詞解釋")
    clean_tokens = filter_tokens(doc)
    num_pattern = re.compile(r"[0-9]")
    clean_tokens = [
        tok for tok in clean_tokens if not num_pattern.search(tok.lemma_)]
    selected_pos = ["VERB", "NOUN", "ADJ", "ADV"]
    clean_tokens = [tok for tok in clean_tokens if tok.pos_ in selected_pos]
    tokens_lemma_pos = [tok.lemma_ + " | " + tok.pos_ for tok in clean_tokens]
    vocab = list(set(tokens_lemma_pos))
    if vocab:
        selected_words = st.multiselect("請選擇要查詢的單詞: ", vocab, vocab[0:3])
        for w in selected_words:
            word_pos = w.split("|")
            word = word_pos[0].strip()
            pos = word_pos[1].strip()
            st.write(f"### {w}")

if morphology:
    st.markdown("## 詞形變化")
    # Collect inflected forms
    inflected_forms = [tok for tok in doc if tok.text.lower()
                       != tok.lemma_.lower()]
    if inflected_forms:
        create_de_df(inflected_forms)

if ner_viz:
    ner_labels = nlp.get_pipe("ner").labels
    visualize_ner(doc, labels=ner_labels, show_table=False, title="命名實體")

if tok_table:
    visualize_tokens(doc, attrs=["text", "pos_",
                     "tag_", "dep_", "head"], title="斷詞特徵")
