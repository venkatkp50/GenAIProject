from text_highlighter import text_highlighter
import streamlit as st

# Basic usage
def highlightText(text,label):
    result = text_highlighter(text=text, labels=[(label, "red")],)


    return result

# # Show the results (as a list)
# st.write(result)