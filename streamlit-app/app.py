import streamlit as st
import random
import time
from generator import load_models, run_model
from utils import change_context, get_top_k, augment_prompt



st.title("askTI")

# Initialize chat history and model
if "messages" not in st.session_state:
    st.session_state.messages = []

if "model" not in st.session_state:
    model, tokenizer = load_models(model_path = 'mistral_local')
    st.session_state.model = model
    st.session_state.tokenizer = tokenizer
    
# # Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("How may I help you?"):
    # Add user message to chat history
    context_list, distances = get_top_k(prompt)
    #print(context_list, distances)
    st.session_state.messages.append({"role": "user", 
                                        "content": prompt, 
                                        })
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    #st.write(st.session_state['messages'])

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        augmented_prompt = augment_prompt(prompt, 
                                         context_list[0])
        #print('AUGMENTED PROMPT :', augmented_prompt)
        ques, ans = run_model(st.session_state.model, 
                              st.session_state.tokenizer,
                             #st.session_state.messages[-1]["content"]
                              augmented_prompt)
        assistant_response = ans#'TEMPLATE DATA'
        # Simulate stream of response with milliseconds delay
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response,'context_list':context_list, 
                                        'distances': distances})

    
        