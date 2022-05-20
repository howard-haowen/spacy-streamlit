from collections import Counter
from dragonmapper import hanzi, transcriptions
import jieba
import pandas as pd
import plotly.express as px
import re
import requests 
import spacy
from spacy_streamlit import visualize_ner, visualize_tokens
#from spacy.language import Language
from spacy.tokens import Doc
import streamlit as st

# Global variables
ZH_TEXT = """（中央社）迎接虎年到來，台北101今天表示，即日起推出「虎年新春燈光秀」，將持續至2月5日，每晚6時至10時，除整點會有報時燈光變化外，每15分鐘還會有3分鐘的燈光秀。台北101下午透過新聞稿表示，今年特別設計「虎年新春燈光秀」，從今晚開始閃耀台北天際線，一直延續至2月5日，共7天。"""
DESCRIPTION = "AI模型輔助語言學習"
TOK_SEP = " | "
PUNCT_SYM = ["PUNCT", "SYM"]

# External API callers
def moedict_caller(word):
    st.write(f"### {word}")
    req = requests.get(f"https://www.moedict.tw/uni/{word}.json")
    try:
        definitions = req.json().get('heteronyms')[0].get('definitions')
        df = pd.DataFrame(definitions)
        df.fillna("---", inplace=True)
        if 'example' not in df.columns:
            df['example'] = '---'
        if 'synonyms' not in df.columns:
            df['synonyms'] = '---' 
        if 'antonyms' not in df.columns:
            df['antonyms'] = '---' 
        cols = ['def', 'example', 'synonyms', 'antonyms']
        df = df[cols]
        df.rename(columns={
            'def': '解釋',
            'example': '例句',
            'synonyms': '同義詞',
            'antonyms': '反義詞',
        }, inplace=True)
        with st.expander("點擊 + 查看結果"):
            st.table(df)
    except:
        st.write("查無結果")
            
# Custom tokenizer class
class JiebaTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = jieba.cut(text) # returns a generator
        tokens = list(words) # convert the genetator to a list
        spaces = [False] * len(tokens)
        doc = Doc(self.vocab, words=tokens, spaces=spaces)
        return doc
    
# Utility functions
def filter_tokens(doc):
    clean_tokens = [tok for tok in doc if tok.pos_ not in PUNCT_SYM]
    clean_tokens = (
        [tok for tok in clean_tokens if 
         not tok.like_email and 
         not tok.like_num and 
         not tok.like_url and 
         not tok.is_space]
    )
    return clean_tokens

def get_vocab(doc):
    clean_tokens = filter_tokens(doc)
    alphanum_pattern = re.compile(r"[a-zA-Z0-9]")
    clean_tokens_text = [tok.text for tok in clean_tokens if not alphanum_pattern.search(tok.text)]
    vocab = list(set(clean_tokens_text))
    return vocab

def get_counter(doc):
    clean_tokens = filter_tokens(doc)
    tokens = [token.text for token in clean_tokens]
    counter = Counter(tokens)
    return counter

def get_freq_fig(doc):
    counter = get_counter(doc)
    counter_df = (
        pd.DataFrame.from_dict(counter, orient='index').
        reset_index().
        rename(columns={
            0: 'count', 
            'index': 'word'
            }).
        sort_values(by='count', ascending=False)
        )
    fig = px.bar(counter_df, x='word', y='count')
    return fig

def get_level_pie(tocfl_result):
    level = tocfl_result['詞條分級'].value_counts()
    fig = px.pie(tocfl_result, 
                values=level.values, 
                names=level.index, 
                title='詞彙分級圓餅圖')
    return fig

@st.cache
def load_tocfl_table(filename="./tocfl_wordlist.csv"):
    table = pd.read_csv(filename)
    cols = "詞彙 漢語拼音 注音 任務領域 詞條分級".split()
    table = table[cols]
    return table
       
# Page setting
st.set_page_config(
    page_icon="🤠",
    layout="wide",
    initial_sidebar_state="auto",
)

# Choose a language and select functions
st.markdown(f"# {DESCRIPTION}") 

# Load the model
nlp = spacy.load('zh_core_web_sm')
          
# Merge entity spans to tokens
# nlp.add_pipe("merge_entities") 

# Select a tokenizer if the Chinese model is chosen
selected_tokenizer = st.radio("請選擇斷詞模型", ["jieba-TW", "spaCy"])
if selected_tokenizer == "jieba-TW":
    nlp.tokenizer = JiebaTokenizer(nlp.vocab)
default_text = ZH_TEXT

st.markdown("## 待分析文本")     
st.info("請在下面的文字框輸入文本並按下Ctrl + Enter以更新分析結果")
text = st.text_area("",  default_text, height=200)
doc = nlp(text)
st.markdown("---")

# keywords_extraction = st.sidebar.checkbox("關鍵詞分析", False) # YAKE doesn't work for Chinese texts
st.info("請勾選以下至少一項功能")
analyzed_text = st.checkbox("增強文本", True)
defs_examples = st.checkbox("單詞解析", True)
# morphology = st.sidebar.checkbox("詞形變化", True)
freq_count = st.checkbox("詞頻統計", True)
ner_viz = st.checkbox("命名實體", True)
tok_table = st.checkbox("斷詞特徵", False)

if analyzed_text:
    st.markdown("## 增強文本") 
    pronunciation = st.radio("請選擇輔助發音類型", ["漢語拼音", "注音符號", "國際音標"])
    for idx, sent in enumerate(doc.sents):
        tokens_text = [tok.text for tok in sent if tok.pos_ not in PUNCT_SYM]
        pinyins = [hanzi.to_pinyin(word) for word in tokens_text]
        sounds = pinyins
        if pronunciation == "注音符號":
            zhuyins = [transcriptions.pinyin_to_zhuyin(word) for word in pinyins]
            sounds = zhuyins
        elif pronunciation == "國際音標":
            ipas = [transcriptions.pinyin_to_ipa(word) for word in pinyins]
            sounds = ipas

        display = []
        for text, sound in zip(tokens_text, sounds):
            res = f"{text} [{sound}]"
            display.append(res)
        if display:
            display_text = TOK_SEP.join(display)
            st.write(f"{idx+1} >>> {display_text}")
        else:
            st.write(f"{idx+1} >>> EMPTY LINE")

if defs_examples:
    st.markdown("## 單詞解析")
    vocab = get_vocab(doc)
    if vocab:
        tocfl_table = load_tocfl_table()
        filt = tocfl_table['詞彙'].isin(vocab)
        tocfl_res = tocfl_table[filt]
        st.markdown("### 華語詞彙分級")
        fig = get_level_pie(tocfl_res)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("點擊 + 查看結果"):
            st.table(tocfl_res)
        st.markdown("---")
        st.markdown("### 單詞解釋與例句")
        selected_words = st.multiselect("請選擇要查詢的單詞: ", vocab, vocab[-1])
        for w in selected_words:
            moedict_caller(w)                        

if freq_count:  
    st.markdown("## 詞頻統計")  
    counter = get_counter(doc)
    topK = st.slider('請選擇前K個高頻詞', 1, len(counter), 5)
    most_common = counter.most_common(topK)
    st.write(most_common)
    st.markdown("---")

    fig = get_freq_fig(doc)
    st.plotly_chart(fig, use_container_width=True)

if ner_viz:
    ner_labels = nlp.get_pipe("ner").labels
    visualize_ner(doc, labels=ner_labels, show_table=False, title="命名實體")
    
if tok_table:
    visualize_tokens(doc, attrs=["text", "pos_", "tag_", "dep_", "head"], title="斷詞特徵")