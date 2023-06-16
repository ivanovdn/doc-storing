prompt_template = """Extract the key facts out of this text. Don't include opinions.

{text}

"""


refine_template = (
    "Your job is to produce a final summary\n"
    "We have provided an existing summary up to a certain point: {existing_answer}\n"
    "We have the opportunity to refine the existing summary"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{text}\n"
    "------------\n"
    "{query}"
)


default_summary_query = """
Pay attention to dates, addresses and named enteties.
Give each fact a number and keep them short sentences.
"""
