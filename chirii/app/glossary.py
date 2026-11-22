"""
Indian everyday-terms glossary (S4d): a curated list, used as the facts for general-knowledge questions about these
terms, so the model can't get them wrong ("UPI stands for United Parcel Service" in run (e3)). Plain, general
descriptions; rules and limits change, so none are given here.
"""

import re
from typing import Dict, List, Optional, Tuple

GLOSSARY: Dict[str, Tuple[str, str]] = {
    "UPI": ("Unified Payments Interface", "a system for sending money instantly from a mobile phone, using a UPI ID or a QR code"),
    "GST": ("Goods and Services Tax", "the tax charged on most goods and services sold in India"),
    "PAN": ("Permanent Account Number", "a 10-character ID from the Income Tax Department, needed for tax returns and many financial transactions"),
    "Aadhaar": ("Aadhaar", "a 12-digit identity number issued by UIDAI, linked to a person's fingerprints and photo"),
    "KYC": ("Know Your Customer", "the identity and address check that banks and other financial firms must do for their customers"),
    "EMI": ("Equated Monthly Instalment", "the fixed amount paid every month to repay a loan, covering both interest and principal"),
    "IFSC": ("Indian Financial System Code", "an 11-character code that identifies a bank branch for online transfers"),
    "NEFT": ("National Electronic Funds Transfer", "a bank-to-bank transfer settled in batches through the day"),
    "RTGS": ("Real Time Gross Settlement", "a bank-to-bank transfer settled immediately, one by one, meant for large amounts"),
    "IMPS": ("Immediate Payment Service", "an instant bank transfer available at any time, including holidays"),
    "TDS": ("Tax Deducted at Source", "income tax taken out by the payer (such as an employer or bank) before paying you"),
    "TCS": ("Tax Collected at Source", "tax collected by the seller from the buyer on certain sales"),
    "ITR": ("Income Tax Return", "the yearly form filed with the Income Tax Department to report income and tax paid"),
    "LIC": ("Life Insurance Corporation of India", "India's large government-owned life insurance company"),
    "PF": ("Provident Fund", "a retirement savings fund to which an employee and employer both contribute"),
    "EPF": ("Employees' Provident Fund", "the provident fund for salaried employees, run by the EPFO"),
    "EPFO": ("Employees' Provident Fund Organisation", "the government body that manages the Employees' Provident Fund"),
    "PPF": ("Public Provident Fund", "a long-term government-backed savings scheme with tax benefits"),
    "UAN": ("Universal Account Number", "a 12-digit number that links all of an employee's provident fund accounts"),
    "FD": ("Fixed Deposit", "money kept with a bank for a fixed period at a fixed interest rate"),
    "RD": ("Recurring Deposit", "a deposit where a fixed amount is saved every month for a set period"),
    "CIBIL": ("CIBIL score", "a credit score from 300 to 900 from TransUnion CIBIL, showing how reliably a person repays loans"),
    "OTP": ("One-Time Password", "a short code sent by SMS or email, valid for a single login or payment"),
    "ATM": ("Automated Teller Machine", "a machine for withdrawing cash and other basic banking"),
    "NACH": ("National Automated Clearing House", "the system banks use for automatic repeated debits such as EMIs and SIPs"),
    "ECS": ("Electronic Clearing Service", "an older system for automatic repeated bank payments, now mostly replaced by NACH"),
    "SIP": ("Systematic Investment Plan", "investing a fixed amount in a mutual fund at regular intervals, usually monthly"),
    "NAV": ("Net Asset Value", "the price of one unit of a mutual fund"),
    "NPS": ("National Pension System", "a government-backed voluntary retirement savings scheme"),
    "HUF": ("Hindu Undivided Family", "a family treated as a separate taxpayer under Indian tax law"),
    "GSTIN": ("GST Identification Number", "the 15-character registration number of a business under GST"),
    "MRP": ("Maximum Retail Price", "the highest price a product can be sold for, printed on the pack"),
    "DTH": ("Direct-to-Home", "satellite TV received through a small dish at home"),
    "FASTag": ("FASTag", "a sticker on a vehicle's windscreen used to pay highway tolls automatically"),
    "RTO": ("Regional Transport Office", "the government office for driving licences and vehicle registration"),
    "PUC": ("Pollution Under Control certificate", "a certificate showing a vehicle's exhaust is within limits"),
    "PNR": ("Passenger Name Record", "the 10-digit number that identifies a train ticket booking"),
    "IRCTC": ("Indian Railway Catering and Tourism Corporation", "the railways' company for online ticket booking and catering"),
    "Tatkal": ("Tatkal", "a train ticket booking quota for urgent travel, opened a day before the journey at a higher fare"),
    "Ration card": ("Ration card", "a state government card for buying subsidised food grains through the public distribution system"),
    "Voter ID": ("Voter ID (EPIC)", "the Election Commission's identity card used for voting"),
    "DigiLocker": ("DigiLocker", "a government app for keeping official documents such as a driving licence digitally"),
    "Ayushman Bharat": ("Ayushman Bharat (PM-JAY)", "a government health insurance scheme for eligible families"),
    "BHIM": ("Bharat Interface for Money", "the government's UPI payments app"),
    "RBI": ("Reserve Bank of India", "India's central bank, which regulates banks and issues currency"),
    "SEBI": ("Securities and Exchange Board of India", "the regulator for the stock market and mutual funds"),
    "IRDAI": ("Insurance Regulatory and Development Authority of India", "the regulator for insurance companies"),
    "Form 16": ("Form 16", "the certificate an employer gives each year showing salary paid and tax deducted"),
    "26AS": ("Form 26AS", "the yearly statement of tax deducted or paid against a PAN"),
    "Challan": ("Challan", "a form or receipt for paying a fee, fine or tax to the government"),
    "Stamp duty": ("Stamp duty", "the state tax paid when registering documents such as a property sale"),
}
ALIASES = {"aadhar": "Aadhaar", "ration": "Ration card", "voter": "Voter ID", "fastag": "FASTag", "form16": "Form 16"}
ASKS_MEANING = re.compile(r"\b(what is|what's|what are|what does|meaning of|stand for|stands for|full form|explain|difference between|"
                          r"tell me about|what do you mean by)\b", re.I)


def terms_in(question: str) -> List[str]:
    found = []
    for term in GLOSSARY:
        pattern = rf"\b{re.escape(term)}\b"
        flags = 0 if term.isupper() and len(term) <= 5 else re.I       # short abbreviations must be in capitals
        if re.search(pattern, question, flags) and term not in found:
            found.append(term)
    for alias, term in ALIASES.items():
        if re.search(rf"\b{alias}\b", question, re.I) and term not in found:
            found.append(term)
    return found


def glossary_answer(question: str) -> Optional[str]:
    """An answer from the glossary if the question asks what one or more of its terms mean."""
    terms = terms_in(question)
    if not terms or not ASKS_MEANING.search(question):
        return None
    lines = []
    for term in terms[:3]:
        full, what = GLOSSARY[term]
        lines.append(f"{term} ({full}) is {what}." if full != term else f"{term} is {what}.")
    return " ".join(lines)
