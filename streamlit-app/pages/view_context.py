import streamlit as st
import random
import time
from generator import load_models, run_model
from utils import change_context, get_top_k, augment_prompt

st.title('Top k Contexts for your question.')

if 'messages' not in st.session_state or st.session_state['messages']==[]:
    st.write('No contexts available')
else:
    contexts = st.session_state.messages[-1]['context_list']
    scores = st.session_state.messages[-1]['distances']

    for i,(score, context) in enumerate(zip(scores,contexts)):
        st.metric(label="L1 Distance", value=round(score,ndigits = 5) )
        st.container(border = True, ).write(context)
        st.button(key = f'use_as_context_{i}',
                  label='Change Context',
                    on_click= change_context,
                    help = 'Click to use above context for generation')
        