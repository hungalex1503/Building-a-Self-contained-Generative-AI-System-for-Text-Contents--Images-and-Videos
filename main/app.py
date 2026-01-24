import streamlit as st
import ollama

# ======================================================
# CONFIGURATION
# ======================================================
# Model hỗ trợ Vision (NHỚ: ollama pull llava-phi3)
MODEL_NAME = "phi3:mini"


# ======================================================
# BACKEND CLASS
# ======================================================
class ContentGenerator:
    def __init__(self):
        self.model = MODEL_NAME

    def generate_text_stream(self, prompt, image_bytes=None, context_history=None):
        """
        Generate streaming response from Ollama.
        Supports optional image input.
        """
        try:
            # User message
            user_message = {
                "role": "user",
                "content": prompt
            }

            # Attach image if provided
            if image_bytes:
                user_message["images"] = [image_bytes]

            # Prepare messages
            messages = context_history if context_history else []
            messages.append(user_message)

            # Stream from Ollama
            stream = ollama.chat(
                model=self.model,
                messages=messages,
                stream=True,
            )

            for chunk in stream:
                yield chunk["message"]["content"]

        except Exception as e:
            yield f"Error: {str(e)}"


# ======================================================
# FRONTEND UI
# ======================================================
def main():
    st.set_page_config(
        page_title="Local AI Marketer (Multimodal)",
        layout="wide"
    )

    # Init chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    uploaded_file = None

    # ==================================================
    # SIDEBAR
    # ==================================================
    with st.sidebar:
        st.header("⚙️ Settings")

        # -------- TODO 1: Persona Builder --------
        custom_persona = st.text_area(
            "Define AI Persona (System Prompt):",
            value="You are an expert marketing copywriter. Create persuasive, catchy, and professional content.",
            height=140
        )

        st.markdown("---")

        # -------- TODO 2: Clear & Save History --------
        st.header("🧹 History Management")

        # Clear chat
        if st.button("🔴 Clear Chat"):
            st.session_state.messages = []
            st.rerun()

        # Build chat log text
        chat_log_content = ""
        for msg in st.session_state.messages:
            role = msg["role"].upper()
            content = msg.get("content", "")
            chat_log_content += f"{role}: {content}\n\n"

        # Download history
        st.download_button(
            label="⬇️ Save History as TXT",
            data=chat_log_content,
            file_name="chat_log.txt",
            mime="text/plain",
            disabled=not bool(st.session_state.messages)
        )

        st.markdown("---")

        # -------- Image Upload --------
        st.header("🖼 Image Analysis")
        uploaded_file = st.file_uploader(
            "Upload an image for context",
            type=["png", "jpg", "jpeg"]
        )

        if uploaded_file:
            st.image(uploaded_file, caption="Preview", use_column_width=True)

        st.markdown("---")
        st.caption(f"Running locally on **{MODEL_NAME}**")

    # ==================================================
    # MAIN UI
    # ==================================================
    st.title("🧠 Local GenAI Workspace (Multimodal)")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Ask about the image or just chat..."):

        # Save & show user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepare API messages
        api_messages = []

        # Insert custom persona
        if custom_persona and custom_persona.strip():
            api_messages.append({
                "role": "system",
                "content": custom_persona
            })

        # Add previous messages (text only)
        for msg in st.session_state.messages[:-1]:
            if msg["role"] != "system":
                api_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        # Generate AI response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            generator = ContentGenerator()

            img_data = uploaded_file.getvalue() if uploaded_file else None

            for chunk in generator.generate_text_stream(
                prompt,
                image_bytes=img_data,
                context_history=api_messages
            ):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

        # Save assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response
        })


# ======================================================
# ENTRY POINT
# ======================================================
if __name__ == "__main__":
    main()
