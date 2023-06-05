import streamlit as st
import pandas as pd
import numpy as np

# Simple Text
st.header('Simple Text')
st.write('Here is some simple text')

# Latex
st.header('LaTeX Equations')
st.write("""Here is a simple equation using LaTeX
$$
ax^2 + bx + c
$$
""")

# Markdown
st.write('## Markdown')
st.write('Here is some more **markdown** text. *And here is some more in italics*')

# Emojis
st.header('Emojis')
st.write('And an emoji or two :smile: :thumbsup:')

# Calculations
st.header('Calculations')
a = 3
b = 3
st.write('And calculations:', a + b )

# Dataframes
st.header('Dataframes')
arr_data = np.random.default_rng().uniform(0, 100, size=(5,5))
df = pd.DataFrame(arr_data, columns=list('ABCDE'))

st.write(df)