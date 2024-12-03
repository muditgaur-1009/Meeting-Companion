import queue
import threading
import sounddevice as sd
import numpy as np
import torch
import chromadb
import google.generativeai as genai
from tavily import TavilyClient
import gradio as gr
import soundfile as sf
from faster_whisper import WhisperModel
from typing import List, Dict, Any
import concurrent.futures
import os

class AgenticRAGSystem:
    def __init__(self):
        # LLM Configuration
        genai.configure(api_key="AIzaSyB1luPxiGaDFNYazTFZMWlwz1Qrg_11Vn8")
        self.llm = genai.GenerativeModel('gemini-1.5-pro')
        
        # Web Search Configuration
        self.tavily = TavilyClient(api_key="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")
        
        # Vector Database Configuration
        self.chroma_client = chromadb.Client()
        try:
            self.collection = self.chroma_client.get_collection("knowledge_base")
        except:
            self.collection = self.chroma_client.create_collection("knowledge_base")

    def extract_implicit_question(self, transcription: str) -> str:
        """
        Advanced implicit question extraction using Chain of Thought
        """
        cot_prompt = f"""Carefully analyze the following transcription to extract the most meaningful implicit question:

Transcription: '{transcription}'

Comprehensive Analysis Process:
1. Identify Core Topics
   - What are the main subjects discussed?
   - Are there any unclear or ambiguous statements?

2. Knowledge Gap Identification
   - What specific information seems to be missing?
   - What would someone want to know more about after hearing this?

3. Question Formulation
   - Transform the implied information need into a precise, answerable question
   - Ensure the question is specific and focused

4. Prioritization
   - Select the most significant question that represents the key insight or curiosity in the transcription

Output Guidelines:
- Provide a single, clear, concise question
- The question should be directly derivable from the transcription
- Avoid broad or overly general questions
- Focus on extracting the most nuanced and meaningful query"""

        try:
            response = self.llm.generate_content(cot_prompt)
            implicit_question = response.text.strip()
            
            # Additional validation to ensure a meaningful question
            if len(implicit_question) < 10 or len(implicit_question) > 200:
                return "What key information is missing from this context?"
            
            return implicit_question
        except Exception as e:
            print(f"Question extraction error: {e}")
            return "What key information is missing from this context?"

    def decision_making_agent(self, question: str, transcription: str) -> str:
        """
        Advanced tool-calling agent with multi-dimensional decision making
        """
        decision_prompt = f"""Determine the OPTIMAL retrieval strategy for this question:

Question: '{question}'
Original Context: '{transcription}'

Comprehensive Decision Matrix:

1. Information Complexity
   - How specialized is the knowledge required?
   - Does it need domain-specific expertise?

2. Temporal Relevance
   - Is this about current events?
   - Does it require up-to-date information?

3. Source Reliability
   - Can internal knowledge (RAG) sufficiently answer?
   - Might external web sources provide better insight?

4. Query Characteristics
   - Factual or analytical?
   - Requires deep explanation or quick facts?

Retrieval Strategy Options:
a) RAG: For deep, specialized, context-rich queries
b) WebSearch: For current events, broad research, recent information
c) DirectLLM: For general knowledge, interpretative questions

Decision Criteria Weights:
- Domain Specificity: High
- Temporal Sensitivity: Moderate
- Complexity: High

Provide ONLY the strategy name: RAG, WebSearch, or DirectLLM

Reasoning should demonstrate a nuanced understanding of the query's nature."""

        try:
            strategy_response = self.llm.generate_content(decision_prompt)
            strategy = strategy_response.text.strip()
            
            valid_strategies = ['RAG', 'WebSearch', 'DirectLLM']
            return strategy if strategy in valid_strategies else 'DirectLLM'
        
        except Exception as e:
            print(f"Strategy decision error: {e}")
            return 'DirectLLM'

    def rag_retrieval(self, question: str) -> str:
        """
        Enhanced RAG Retrieval with more intelligent context matching
        """
        try:
            # More sophisticated retrieval using semantic similarity
            retrieved_docs = self.collection.query(
                query_texts=[question],
                n_results=3
            )
            
            context = "\n".join(retrieved_docs.get('documents', []))
            
            rag_prompt = f"""Synthesize a comprehensive answer using the following context:

Precise Question: {question}
Retrieved Context: {context}

Analysis Guidelines:
1. Directly address the specific question
2. Use retrieved context to provide a well-supported answer
3. If context is insufficient, clearly explain what additional information is needed
4. Prioritize accuracy and relevance over verbosity"""
            
            response = self.llm.generate_content(rag_prompt)
            return response.text
        
        except Exception as e:
            print(f"RAG retrieval error: {e}")
            return "Specialized retrieval failed. Switching to alternative strategy."

    def web_search(self, question: str) -> str:
        """
        Precise web search with advanced result synthesis
        """
        try:
            # Use the extracted implicit question for more precise search
            search_results = self.tavily.search(query=question, max_results=3)
            
            synthesis_prompt = f"""Synthesize a comprehensive, authoritative answer from these web search results:

Precise Search Query: '{question}'
Search Results: {search_results}

Synthesis Instructions:
1. Extract core factual information
2. Cross-reference multiple sources
3. Provide a clear, concise, and informative response
4. Include key insights from the search results
5. Maintain academic rigor in presentation"""
            
            summary = self.llm.generate_content(synthesis_prompt)
            return summary.text
        
        except Exception as e:
            print(f"Web search error: {e}")
            return "Web search encountered challenges. Falling back to direct knowledge retrieval."

    def direct_llm_response(self, question: str) -> str:
        """
        Intelligent direct LLM response generation
        """
        try:
            comprehensive_prompt = f"""Provide a thoughtful, comprehensive response to the following query:

Question: {question}

Response Guidelines:
1. Draw from broad knowledge base
2. Provide nuanced, multi-perspective insights
3. Maintain clarity and intellectual depth
4. If uncertain, acknowledge limitations
5. Prioritize informative and balanced explanation"""
            
            response = self.llm.generate_content(comprehensive_prompt)
            return response.text
        
        except Exception as e:
            print(f"Direct LLM response error: {e}")
            return "Unable to generate a comprehensive response at this time."

    def process_transcription(self, transcription: str) -> Dict[str, str]:
        """
        Orchestration method with enhanced decision-making
        """
        # Step 1: Extract Implicit Question
        implicit_question = self.extract_implicit_question(transcription)
        
        # Step 2: Decide Retrieval Strategy
        strategy = self.decision_making_agent(implicit_question, transcription)
        
        # Step 3: Execute Chosen Strategy
        if strategy == 'RAG':
            answer = self.rag_retrieval(implicit_question)
        elif strategy == 'WebSearch':
            answer = self.web_search(implicit_question)
        else:
            answer = self.direct_llm_response(implicit_question)
        
        return {
            'transcription': transcription,
            'question': implicit_question,
            'strategy': strategy,
            'answer': answer
        }


class AudioCapture:
    def __init__(self, sample_rate=16000, channels=1, device=None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.stream = None
        self.recording_thread = None
        self.stop_event = threading.Event()

    def start_recording(self):
        self.is_recording = True
        self.stop_event.clear()
        
        def recording_worker():
            with sd.InputStream(
                samplerate=self.sample_rate, 
                channels=self.channels,
                callback=self._audio_callback,
                device=self.device
            ) as stream:
                while not self.stop_event.is_set():
                    sd.sleep(100)

        self.recording_thread = threading.Thread(target=recording_worker)
        self.recording_thread.start()

    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        if self.is_recording:
            self.audio_queue.put(indata.copy())

    def stop_recording(self):
        self.is_recording = False
        self.stop_event.set()
        if self.recording_thread:
            self.recording_thread.join()

    def get_audio_chunks(self):
        chunks = []
        while not self.audio_queue.empty():
            chunks.append(self.audio_queue.get())
        return np.concatenate(chunks) if chunks else np.array([])

class TranscriptionHandler:
    def __init__(self):
        # Initialize Faster Whisper model with medium English model
        self.stt_model = WhisperModel(
            "medium.en", 
            device="cuda",  # Use "cpu" if no GPU available
            compute_type="float16"
        )
        
        self.transcription_queue = queue.Queue()
        self.transcription_history = []
        self.transcription_lock = threading.Lock()

    def transcribe_audio(self, audio_data):
        # Write audio data to a temporary wav file
        sf.write('temp_audio.wav', audio_data, 16000)
        
        try:
            # Transcribe using Faster Whisper
            segments, info = self.stt_model.transcribe(
                'temp_audio.wav', 
                beam_size=5,
                language='en'
            )
            
            # Combine transcribed segments
            transcription = " ".join([segment.text for segment in segments])
            
            if transcription.strip():
                with self.transcription_lock:
                    self.transcription_queue.put(transcription)
                    self.transcription_history.append(transcription)
            
            return transcription
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""

class VoiceAssistantApp:
    def __init__(self):
        self.audio_capture = AudioCapture()
        self.transcription_handler = TranscriptionHandler()
        self.rag_system = AgenticRAGSystem()
        
        self.transcriptions = []
        self.questions = []
        self.answers = []
        self.strategies = []
        
        # Thread-safe list management
        self.list_lock = threading.Lock()

    def process_audio(self):
        audio_chunk = self.audio_capture.get_audio_chunks()
        
        if len(audio_chunk) > 0:
            # Use thread pool for concurrent processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # Submit transcription task
                transcribe_future = executor.submit(
                    self.transcription_handler.transcribe_audio, 
                    audio_chunk
                )
                transcription = transcribe_future.result()
                
                if transcription:
                    # Process transcription through Agentic RAG
                    rag_result = self.rag_system.process_transcription(transcription)
                    
                    # Thread-safe list update
                    with self.list_lock:
                        self.transcriptions.append(transcription)
                        self.questions.append(rag_result['question'])
                        self.answers.append(rag_result['answer'])
                        self.strategies.append(rag_result['strategy'])
                        
                        # Limit history
                        self.transcriptions = self.transcriptions[-10:]
                        self.questions = self.questions[-10:]
                        self.answers = self.answers[-10:]
                        self.strategies = self.strategies[-10:]
                    
                    return (
                        "\n".join(self.transcriptions), 
                        "\n".join(self.questions), 
                        "\n".join(self.answers),
                        "\n".join(self.strategies)
                    )
        
        return "", "", "", ""

    def create_gradio_interface(self):
        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column():
                    transcription_output = gr.Textbox(label="Transcriptions")
                with gr.Column():
                    question_output = gr.Textbox(label="Identified Questions")
                    answer_output = gr.Textbox(label="Answers")
                    strategy_output = gr.Textbox(label="Retrieval Strategy")
                
            start_btn = gr.Button("Start Recording")
            start_btn.click(
                fn=self.audio_capture.start_recording, 
                inputs=None, 
                outputs=None
            )
            
            stop_btn = gr.Button("Stop Recording")
            stop_btn.click(
                fn=self.audio_capture.stop_recording, 
                inputs=None, 
                outputs=None
            )
            
            process_btn = gr.Button("Process Audio")
            process_btn.click(
                fn=self.process_audio, 
                inputs=None, 
                outputs=[transcription_output, question_output, answer_output, strategy_output]
            )

        demo.launch()

# Run the application
if __name__ == "__main__":
    app = VoiceAssistantApp()
    app.create_gradio_interface()