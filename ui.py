import streamlit as st
from model import app, StreamlitCallbackHandler

def main():
    # Sidebar for conversation management
    with st.sidebar:
        st.header("Conversations")
        
        # Button to start a new conversation
        if st.button("New Conversation"):
            st.session_state.conversation_id = len(st.session_state.get("conversations", []))
            st.session_state.state = {
                "messages": [],
                "context": "",
                "question": ""
            }
            st.rerun()

        # Display existing conversations
        if "conversations" not in st.session_state:
            st.session_state.conversations = []

        for i, conv in enumerate(st.session_state.conversations):
            if st.button(f"Conversation {i + 1}", key=f"conv_{i}"):
                st.session_state.conversation_id = i
                st.session_state.state = conv
                st.rerun()

    # Initialize conversation state if not exists
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = 0

    if "state" not in st.session_state:
        st.session_state.state = {
            "messages": [],
            "context": "",
            "question": ""
        }

    # Display current conversation
    st.title(f"Bhagavad Gita RAG Chatbot - Conversation {st.session_state.conversation_id + 1}")

    for message in st.session_state.state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask your question about the Bhagavad Gita:"):
        st.session_state.state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            stream_container = st.empty()
            with stream_container:
                callback = StreamlitCallbackHandler(stream_container)

                new_state = app.invoke(
                    st.session_state.state,
                    {"callbacks": [callback]}
                )

        st.session_state.state = new_state

        # Update conversations list
        if len(st.session_state.conversations) <= st.session_state.conversation_id:
            st.session_state.conversations.append(st.session_state.state)
        else:
            st.session_state.conversations[st.session_state.conversation_id] = st.session_state.state

if __name__ == '__main__':
    main()