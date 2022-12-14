import pandas as pd
import re
import requests 
import spacy
from spacy_streamlit import visualize_ner, visualize_tokens
#from spacy.language import Language
from spacy.tokens import Doc
import spacy_ke
import streamlit as st
from wiktionaryparser import WiktionaryParser

# Global variables
DEFAULT_TEXT = """오늘 날씨 너무 좋다. 1908년 국어연구학회가 창립된 이래 여러 시련에도 불구하고 한글연구의 명맥은 꾸준히 이어졌으며, 한글날 제정, 사전편찬, 맞춤법 제정 등 많은 성과들을 일구어냈다. 광복후 '조선어학회'가 활동을 재개하였고 1949년에 '한글학회'로 개칭되면서 한글 표준화 사업등 많은 노력이 있었다. 그 결과 한글은 한국어를 표기하는 국어로서의 위상을 지키게 되었다."""
DESCRIPTION = "AI模型輔助語言學習：韓文"
TOK_SEP = " | "
MODEL_NAME = "ko_core_news_sm"
API_LOOKUP = {}
MAX_SYM_NUM = 5
WP = WiktionaryParser()
WP.set_default_language("korean")


# External API caller
def free_dict_caller(word):
    try:
        req = WP.fetch(word)
        if len(req[0]["definitions"]) > 0:
            API_LOOKUP[word] = req[0]
        else:
            API_LOOKUP[word] = None
    except:
        API_LOOKUP[word] = None
 
def show_definitions_and_examples(word, pos):
    if pos == "ADJ":
        pos = "ADJECTIVE"
    if pos == "ADV":
        pos = "ADVERB"
    if word not in API_LOOKUP:
        free_dict_caller(word)
    
    result = API_LOOKUP.get(word)
    if result:
        meanings = result.get("definitions")
        if meanings:
            definitions = []
            examples = []
            for meaning in meanings:
                if meaning["partOfSpeech"] == pos.lower():
                    definitions = meaning.get("text")
                    examples = meaning.get("examples")
                    break

            if len(definitions) > 3:
              definitions = definitions[:3]
            if len(examples) > 3:
              examples = examples[:3]

            st.markdown(f" Definitions: ")
            for df in definitions:
              st.markdown(f" - {df}")
            st.markdown(f" Examples: ")
            for ex in examples:
                st.markdown(f" - *{ex}*")
            st.markdown("---")  

    else:
        st.info("Found no matching result on Free Dictionary!")

def get_pron(word, pos):
    if word not in API_LOOKUP:
        free_dict_caller(word)
    
    result = API_LOOKUP.get(word)
    try:
        if result:
            pron = result.get("pronunciations")
            if pron:
                return pron.get("text")[0].split('IPA: ')[1]
    except:
        return ""
        
# Utility functions
def create_kr_df(tokens):
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
      file_name='kr_forms.csv',
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
    kws2scores = {keyword: score for keyword, score in doc._.extract_keywords(n=kw_num)}
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
nlp.add_pipe("yake") # keyword extraction
# nlp.add_pipe("merge_entities") # Merge entity spans to tokens

# Page starts from here
st.markdown("## 待分析文本")     
st.info("請在下面的文字框輸入文本並按下Ctrl + Enter以更新分析結果")
text = st.text_area("",  DEFAULT_TEXT, height=200)
doc = nlp(text)
st.markdown("---")

st.info("請勾選以下至少一項功能")
keywords_extraction = st.checkbox("關鍵詞分析", False)
analyzed_text = st.checkbox("增強文本", True)
defs_examples = st.checkbox("單詞解析", True)
morphology = st.checkbox("詞形變化", False)
ner_viz = st.checkbox("命名實體", True)
tok_table = st.checkbox("斷詞特徵", False)

if keywords_extraction:
    create_kw_section(doc)

if analyzed_text:
    st.markdown("## 分析後文本")     
    for idx, sent in enumerate(doc.sents):
        enriched_sentence = []
        for tok in sent:
            pron = get_pron(tok.text, tok.pos_)
            if pron is None or pron == "":
                enriched_sentence.append(tok.text)
            else:
                enriched_tok = f"{tok.text} (IPA: {pron})"
                enriched_sentence.append(enriched_tok)

        display_text = " ".join(enriched_sentence)
        st.write(f"{idx+1} >>> {display_text}")     

if defs_examples:
    st.markdown("## 單詞解釋與例句")
    clean_tokens = filter_tokens(doc)
    num_pattern = re.compile(r"[0-9]")
    clean_tokens = [tok for tok in clean_tokens if not num_pattern.search(tok.lemma_)]
    selected_pos = ["VERB", "NOUN", "ADJ", "ADV"]
    clean_tokens = [tok for tok in clean_tokens if tok.pos_ in selected_pos]
    tokens_lemma_pos = [re.sub("\+", "", tok.lemma_) + " | " + tok.pos_ for tok in clean_tokens]
    vocab = list(set(tokens_lemma_pos))
    if vocab:
        selected_words = st.multiselect("請選擇要查詢的單詞: ", vocab, vocab[0:3])
        for w in selected_words:
            word_pos = w.split("|")
            word = word_pos[0].strip()
            pos = word_pos[1].strip()
            st.write(f"### {w}")
            with st.expander("點擊 + 檢視結果"):
                show_definitions_and_examples(word, pos)

if morphology:
    st.markdown("## 詞形變化")
    # Collect inflected forms
    inflected_forms = [tok for tok in doc if tok.text.lower() != tok.lemma_.lower()]
    if inflected_forms:
        create_kr_df(inflected_forms)

if ner_viz:
    ner_labels = nlp.get_pipe("ner").labels
    visualize_ner(doc, labels=ner_labels, show_table=False, title="命名實體")

if tok_table:
    visualize_tokens(doc, attrs=["text", "pos_", "tag_", "dep_", "head"], title="斷詞特徵")
