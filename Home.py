import streamlit as st
from streamlit.components.v1 import html


st.set_page_config(
    page_title="Understanding intent",
    page_icon="👋",
)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: black; margin-top: 100px;'>FRAPPE: FRAming, Persuasion, and Propaganda Explorer</h1>", unsafe_allow_html=True)

def nav_page(page_name, timeout_secs=3):
    nav_script = """
        <script type="text/javascript">
            function attempt_nav_page(page_name, start_time, timeout_secs) {
                var links = window.parent.document.getElementsByTagName("a");
                for (var i = 0; i < links.length; i++) {
                    if (links[i].href.toLowerCase().endsWith("/" + page_name.toLowerCase())) {
                        links[i].click();
                        return;
                    }
                }
                var elasped = new Date() - start_time;
                if (elasped < timeout_secs * 1000) {
                    setTimeout(attempt_nav_page, 100, page_name, start_time, timeout_secs);
                } else {
                    alert("Unable to navigate to page '" + page_name + "' after " + timeout_secs + " second(s).");
                }
            }
            window.addEventListener("load", function() {
                attempt_nav_page("%s", new Date(), %d);
            });
        </script>
    """ % (page_name, timeout_secs)
    html(nav_script)


local_css("styles.css")
c4, c1, c3, c2 = st.columns([2,4, 1, 4])

if c1.button("Analyze an article on the fly"):
    nav_page("Article On the Fly")
if c2.button("Visualization of 2M Articles"):
    nav_page("Visualization of 2M Articles")
