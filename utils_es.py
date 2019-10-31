import os
# from tika import parser
# from fpdf import FPDF
import base64
import json
import PyPDF2
from elasticsearch import Elasticsearch

# Convert a pdf to dict which contains 'text': list of text pages, maybe 'title', 'CreationDate', etc
def pdf2text(path):
    text_info = {}
    text_info['text'] = []

    pdf_reader = PyPDF2.PdfFileReader(path, strict=False)
    num_pages = pdf_reader.getNumPages()

    pdf_docinfo = pdf_reader.getDocumentInfo()
    for title, text in pdf_docinfo.items():
        text_info[title] = text

    for page in range(num_pages):
        # get pageObject from the page number
        data = pdf_reader.getPage(page)
        # text extraction
        page_text = data.extractText()
        # store the text to dict
        text_info['text'].append(page_text)
    return text_info

# index a dict (pdf file) to ES
def index2ES(text_info, es, config):
    assert type(text_info) == dict, "param must be a dict"
    # create a JSON string from the dictionary
    json_data = json.dumps(text_info)
    # convert json string to bytes
    bytes_string = json_data.encode('utf-8')
    # convert bytes to base64 encoded string
    encoded_pdf = base64.b64encode(bytes_string)
    encoded_pdf = encoded_pdf.decode("utf-8")
    body_doc = {"data": encoded_pdf}
    # Adds or updates a typed JSON document in a specific index, making it searchable.
    es.index(index=config['index'], doc_type="_doc", id=config['id'],  body=body_doc)

# query a dict from es
def get(es, config):

    # make another Elasticsearch API request to get the indexed PDF
    result = es.get(index=config['index'], doc_type='_doc', id=config['id'])
    print(result)

    # print the data to terminal
    result_data = result["_source"]["data"]
    # print("\nresult_data:", result_data, '-- type:', type(result_data))


    # decode the base64 data (use to [:] to slice off
    # the 'b and ' in the string)
    decoded_pdf = base64.b64decode(result_data[2:-1]).decode("utf-8")
    # print("\ndecoded_pdf:", decoded_pdf)

    # take decoded string and make into JSON object
    json_dict = json.loads(decoded_pdf)
    # print("\njson_str:", json_dict, "\nntype:", type(json_dict))