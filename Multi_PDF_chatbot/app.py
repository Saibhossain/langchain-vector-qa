import streamlit as st
import os
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_ollama import OllamaLLM
from utils import load_and_index_pdfs, extract_images_and_tables

