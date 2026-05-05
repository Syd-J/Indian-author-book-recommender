"""
Streamlit UI for the Indian Author Book Recommender.

    streamlit run app.py
"""
import streamlit as st

from recommender import BookRecommender

st.set_page_config(
    page_title="Indian Author Book Recommender",
    page_icon="📚",
    layout="wide",
)

# Cache the model + index across reruns. ~80MB RAM for MiniLM.
@st.cache_resource
def load_recommender() -> BookRecommender:
    return BookRecommender("data")


rec = load_recommender()

# --- session state ---
if "favourites" not in st.session_state:
    st.session_state.favourites = []
if "active_query" not in st.session_state:
    st.session_state.active_query = ""

# --- sidebar ---
with st.sidebar:
    st.title("📚 Filters")
    st.caption("Leave blank to search the whole library.")
    author_filter = st.text_input("Author contains", "")
    k = st.slider("Number of results", min_value=3, max_value=20, value=8)
    min_score = st.slider("Min similarity", 0.0, 1.0, 0.10, 0.05)

    st.divider()
    st.subheader(f"⭐ Favourites ({len(st.session_state.favourites)})")
    if not st.session_state.favourites:
        st.caption("Your saved books will appear here.")
    else:
        for fav in st.session_state.favourites:
            st.markdown(f"• **{fav['title']}** — {fav['authors']}")
        if st.button("Clear all", use_container_width=True):
            st.session_state.favourites = []
            st.rerun()

    st.divider()
    st.caption(f"Corpus: {len(rec.df)} books")

# --- main ---
st.title("Indian Author Book Recommender")
st.caption("Semantic search over Open Library's Indian-author catalogue. "
           "Tell me what you feel like reading.")

# Quick chips
chips = [
    "partition stories",
    "magical realism",
    "Bengali poetry",
    "modern Indian fiction",
    "mythology retold",
    "rural Indian life",
    "post-colonial",
    "small-town India",
]
chip_cols = st.columns(len(chips))
for i, ch in enumerate(chips):
    if chip_cols[i].button(ch, use_container_width=True, key=f"chip-{i}"):
        st.session_state.active_query = ch
        st.session_state["query_input"] = ch

query = st.text_input(
    "Or describe what you want:",
    placeholder="e.g. coming-of-age in 1970s Calcutta, with poetic prose",
    key="query_input",
)

if query:
    with st.spinner("Searching..."):
        results = rec.search(
            query,
            k=k,
            author_filter=author_filter or None,
            min_score=min_score,
        )

    if not results:
        st.warning("No matches. Try a broader query, lower the min similarity, or clear the author filter.")
    else:
        st.success(f"Top {len(results)} matches for **{query}**")

        # 2-column responsive grid of cards
        for i in range(0, len(results), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j >= len(results):
                    break
                book = results[i + j]
                with col:
                    with st.container(border=True):
                        head = st.columns([1, 3])
                        with head[0]:
                            if book["cover_url"]:
                                st.image(book["cover_url"], width=110)
                            else:
                                st.markdown("###### *(no cover)*")
                        with head[1]:
                            st.markdown(f"### {book['title']}")
                            byline = f"by {book['authors']}" if book["authors"] else ""
                            if book["year"]:
                                byline += f" · {book['year']}"
                            if byline:
                                st.caption(byline)
                            st.caption(f"**Similarity:** `{book['score']:.3f}`")

                        if book["description"]:
                            with st.expander("Description"):
                                desc = book["description"]
                                st.write(desc[:500] + ("..." if len(desc) > 500 else ""))

                        st.markdown(f"**Why?** {rec.justify(query, book)}")

                        save_key = f"fav-{book['work_key']}-{i}-{j}"
                        if st.button("⭐ Save to favourites", key=save_key, use_container_width=True):
                            entry = {"title": book["title"], "authors": book["authors"]}
                            if entry not in st.session_state.favourites:
                                st.session_state.favourites.append(entry)
                                st.toast(f"Saved: {book['title']}", icon="⭐")
                            else:
                                st.toast("Already saved.", icon="ℹ️")
else:
    st.info("Type a query above or click a chip to start.")
    with st.expander("💡 Sample queries to try"):
        st.write(
            "- *Stories about families in post-independence India*\n"
            "- *Mythological retellings with strong female characters*\n"
            "- *Indian author writing in the style of magical realism*\n"
            "- *Coming-of-age in a small village*\n"
            "- *Detective fiction set in Mumbai*\n"
            "- *Poetry about the Bengal famine*\n"
            "- *Satirical novels about Indian bureaucracy*"
        )