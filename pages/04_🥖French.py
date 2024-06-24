import re, pandas as pd
import requests
import spacy
from spacy_streamlit import (
    visualize_ner,
    visualize_tokens,
)
from spacy.language import Language
from spacy.tokens import Doc
import spacy_ke
import streamlit as st

DEFAULT_TEXT = """L'espace de Wikipédia en français est créé le 11 mai 2001. La première version connue de Wikipédia en français voit le jour le 19 mai 2001 à 1 h 05. Les trois premiers articles sont à caractère scientifique et datent du 4 août 2001. Pourtant, c'est le 23 mars 2021, que Wikipédia en français fête officiellement ses 20 ans. Elle atteint un million d'articles le 23 septembre 2010, deux millions le 8 juillet 2018 et compte 2 474 416 articles le 28 novembre 2022. Elle est en 5e position en nombre d'articles, après les éditions en anglais, en cebuano, en allemand et en suédois ; les éditions en cebuano et en suédois étant développées en partie à l'aide d'un bot."""
DESCRIPTION = "AI模型輔助語言學習：法語"
TOK_SEP = " | "
MODEL_NAME = "fr_core_news_sm"
REMAP_POS = {
    "ADV":  "adverb"   ,
    "VERB":  "verb"    ,
    "NOUN":  "noun"    ,
}

from wiktionaryparser import WiktionaryParser

PARSER = WiktionaryParser()
PARSER.set_default_language("french")



if """Load model""":
    MAX_SYM_NUM = 5

    API_LOOKUP = {}
    def free_dict_caller(word):
        if word not in API_LOOKUP:
            try:
                req = PARSER.fetch(word)
                API_LOOKUP[word] = req[0]
            except:
                API_LOOKUP[word] = {}
        return API_LOOKUP[word]

    nlp = spacy.load(MODEL_NAME)
nlp.add_pipe("yake")

st.markdown("## 待分析文本")
st.info("請在下面的文字框輸入文本並按下Ctrl + Enter以更新分析結果")
text = st.text_area("", DEFAULT_TEXT, height=200)
doc = nlp(text)
st.markdown("---")

st.info("請勾選以下至少一項功能")
keywords_extraction = st.checkbox("關鍵詞分析", False)  # F
analyzed_text       = st.checkbox("增強文本",   True)
defs_examples       = st.checkbox("單詞解析",   True)  # F
morphology          = st.checkbox("詞形變化",   False)
ner_viz             = st.checkbox("命名實體",   True)
tok_table           = st.checkbox("斷詞特徵",   False)  # F

if keywords_extraction:
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

    create_kw_section(doc)

if analyzed_text:
    st.markdown("## 分析後文本")
    def get_word_pronunciation(word_lemma):
        try:
            [word] = PARSER.fetch(word_lemma.lower())
            _p = (word["pronunciations"]["text"])
            # pronunciation = [pron for pron in _p if pron.startswith('IPA: ')][0].replace('IPA: ', '')
            # print(_p)
            _ss = [pron for pron in _p if pron.startswith("IPA: ")]
            if len(_ss) == 0:
                return ''
                
            _q = (' '.join(_ss))
            _r = _q.replace("IPA: ", "")
            # print(_r)
            pronunciation = _r
        except:
            return ''
        
        return f" «{pronunciation}»"

    for idx, sent in enumerate(doc.sents):
        enriched_sentence = []
        for tok in sent:
            enriched_sentence.append(
                f"{tok.text}{get_word_pronunciation(tok.text)}"
            )
        #########################################
        display_text = " ".join(enriched_sentence)
        st.write(f"{idx+1} >>> {display_text}")

if defs_examples:
    def filter_tokens(doc):
        clean_tokens = [tok for tok in doc if tok.pos_ not in ["PUNCT", "SYM"]]
        clean_tokens = [tok for tok in clean_tokens if not tok.like_email]
        clean_tokens = [tok for tok in clean_tokens if not tok.like_url]
        clean_tokens = [tok for tok in clean_tokens if not tok.like_num]
        clean_tokens = [tok for tok in clean_tokens if not tok.is_punct]
        clean_tokens = [tok for tok in clean_tokens if not tok.is_space]
        return clean_tokens
    def show_definitions_and_examples(word, pos):
        result = free_dict_caller(word)
        
        if result:
            meanings = result.get('definitions')
            if meanings:
                examples = [
                    meaning['examples']
                    for meaning in meanings
                    if meaning['partOfSpeech'] == REMAP_POS.get(pos)
                ][:3]
                
                for ex in examples[:3]:
                    for _ex in ex[:3]:
                        st.markdown(f" Example: *{_ex}*")
                    st.markdown("---")  
                    
        else:
            st.info("Found no matching result on Wiktionary!")

    st.markdown("## 單詞解釋與例句")
    clean_tokens = filter_tokens(doc)
    num_pattern = re.compile(r"[0-9]")
    clean_tokens = [
        tok for tok in clean_tokens if not num_pattern.search(tok.lemma_)
    ]
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
            with st.expander("點擊 + 檢視結果"):
                show_definitions_and_examples(word, pos)

if morphology:
    def create_eng_df(tokens):
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
          file_name='eng_forms.csv',
          )


    st.markdown("## 詞形變化")
    # Collect inflected forms
    inflected_forms = [tok for tok in doc if tok.text.lower() != tok.lemma_.lower()]
    if inflected_forms:
        create_eng_df(inflected_forms)

if ner_viz:
    ner_labels = nlp.get_pipe("ner").labels
    visualize_ner(doc, labels=ner_labels, show_table=False, title="命名實體")

if tok_table:
    visualize_tokens(doc, attrs=["text", "pos_", "tag_", "dep_", "head"], title="斷詞特徵")
