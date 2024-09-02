import streamlit as st


def display_metadata(query_result):
    """
    Displays the metadata of a query result.
    """
    if query_result:
        st.write("metadata:")

        for entry in query_result:
            st.write("Source:", entry.metadata.get(
                "file_name", "Unknown source").split("/")[-1])

            page = entry.metadata.get("page", 1)
            st.write("Page:", page)

            with st.expander("content", expanded=False):
                st.write(entry.page_content)

            st.write("---")
