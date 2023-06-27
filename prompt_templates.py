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


default_summary = {
    "output_text": """
1. Berkshire Hathaway Inc. will hold its Annual Meeting of Shareholders on May 6, 2023 at the CHI Health Center in Omaha, Nebraska. 2. The meeting will include the election of directors and advisory votes on executive compensation and the frequency of such votes. 3. Shareholders will also act on six proposals and consider any other matters that may arise. 4. The record date for determining voting rights is March 8, 2023. 5. Shareholders can vote over the internet, by telephone, by mail, or at the meeting. 6. A shareholder may request credentials for admission to the meeting. 7. Meeting credentials may be obtained at the meeting by persons identifying themselves as shareholders as of the record date. 8. Possession of a proxy card, a voting information form received from a bank or broker, or a broker's statement showing shares owned on March 8, 2023, along with proper identification will be required for admission. 9. The Proxy Statement and the 2022 Annual Report to the Shareholders are available at www.berkshirehathaway.com/eproxy. 10. The Board of Directors of Berkshire Hathaway Inc. is soliciting proxies for the Annual Meeting of Shareholders. 11. The proxy statement and form were first sent to shareholders on or about March 17, 2023. 12. The record date for the Annual Meeting is March 8, 2023. 13. As of the record date, there were 590,238 shares of Class A Common Stock and 1,298,190,161 shares of Class B Common Stock outstanding and entitled to vote. 14. Each share of Class A Stock is entitled to one vote per share, and each share of Class B Stock is entitled to one-ten-thousandth (1/10,000) of one vote per share on all matters submitted to a vote of shareholders. 15. The Class A Stock and Class B Stock vote together as a single class on the matters described in the proxy statement. 16. Only shareholders of record at the close of business on March 8, 2023, are entitled to vote at the Annual Meeting or any adjournment thereof. 17. The Corporation will reimburse brokerage firms, banks, trustees, and others for their actual out-of-pocket expenses in forwarding proxy material to the beneficial owners of its common stock.
"""
}
