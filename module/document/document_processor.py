"""
Document Processor Module
Xử lý và chunking documents cho hệ thống RAG - Tối ưu cho trading documents
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import hashlib
import time
from datetime import datetime

# Document processing libraries
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    from pptx import Presentation
    PPTX_AVAILABLE = True
except ImportError:
    PPTX_AVAILABLE = False

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Language detection
try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False


@dataclass
class DocumentChunk:
    """Đại diện cho một chunk của tài liệu"""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    source_file: str
    chunk_index: int
    start_char: int
    end_char: int
    chunk_type: str  # 'header', 'content', 'qa_pair', 'list_item'
    language: str = "vi"
    chunk_strategy: str = "unknown"  # header_based, qa_based, size_based, semantic_based
    header_level: int = 0  # Cấp độ header (0 = không phải header)
    qa_count: int = 0  # Số lượng Q&A trong chunk
    token_count: int = 0  # Số lượng tokens (ước tính)


@dataclass
class DocumentInfo:
    """Thông tin tổng quan về tài liệu"""
    file_path: str
    file_type: str
    file_size: int
    total_chunks: int
    language: str
    document_structure: Dict[str, Any]
    processing_time: float
    chunking_strategy: str
    document_type: str  # faq_behavior, faq_training, handbook, presentation


class DocumentProcessor:
    """
    Xử lý và chunking documents thông minh
    Tối ưu cho trading documents với cấu trúc đặc biệt
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Kiểm tra dependencies
        if not any([DOCX_AVAILABLE, PPTX_AVAILABLE, PDF_AVAILABLE]):
            raise ImportError("No document processing library available. Install: pip install python-docx python-pptx PyPDF2")
    
    def process_document(self, file_path: str) -> Tuple[List[DocumentChunk], DocumentInfo]:
        """
        Xử lý document và tạo chunks thông minh
        
        Args:
            file_path: Đường dẫn đến file document
            
        Returns:
            Tuple (chunks, document_info)
        """
        start_time = time.time()
        file_path = Path(file_path)
        
        try:
            print(f"📄 Processing document: {file_path.name}")
            
            # Đọc nội dung document
            content = self._read_document(file_path)
            if not content:
                raise ValueError(f"Empty content from {file_path}")
            
            # Phân tích cấu trúc document
            structure = self._analyze_document_structure(content)
            
            # Xác định loại document
            document_type = self._determine_document_type(file_path.name)
            
            # Tạo chunks thông minh
            chunks = self._create_smart_chunks(content, file_path, structure)
            
            # Tính toán thời gian xử lý
            processing_time = time.time() - start_time
            
            # Tạo document info
            doc_info = DocumentInfo(
                file_path=str(file_path),
                file_type=file_path.suffix.lower(),
                file_size=file_path.stat().st_size,
                total_chunks=len(chunks),
                language=structure.get("detected_language", "vi"),
                document_structure=structure,
                processing_time=processing_time,
                chunking_strategy=structure.get("chunking_strategy", "mixed"),
                document_type=document_type
            )
            
            print(f"✅ Document processed: {len(chunks)} chunks in {processing_time:.2f}s")
            return chunks, doc_info
            
        except Exception as e:
            print(f"❌ Error processing document {file_path}: {e}")
            raise
    
    def _read_document(self, file_path: Path) -> str:
        """Đọc nội dung document dựa trên loại file"""
        file_extension = file_path.suffix.lower()
        
        if file_extension == ".docx" and DOCX_AVAILABLE:
            return self._read_docx(file_path)
        elif file_extension == ".pptx" and PPTX_AVAILABLE:
            return self._read_pptx(file_path)
        elif file_extension == ".pdf" and PDF_AVAILABLE:
            return self._read_pdf(file_path)
        elif file_extension == ".txt":
            return self._read_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    def _read_docx(self, file_path: Path) -> str:
        """Đọc file DOCX"""
        try:
            doc = Document(file_path)
            content = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    content.append(paragraph.text)
            
            return "\n".join(content)
        except Exception as e:
            print(f"❌ Error reading DOCX: {e}")
            return ""
    
    def _read_pptx(self, file_path: Path) -> str:
        """Đọc file PPTX"""
        try:
            prs = Presentation(file_path)
            content = []
            
            for slide in prs.slides:
                slide_content = []
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        slide_content.append(shape.text)
                if slide_content:
                    content.append(" | ".join(slide_content))
            
            return "\n".join(content)
        except Exception as e:
            print(f"❌ Error reading PPTX: {e}")
            return ""
    
    def _read_pdf(self, file_path: Path) -> str:
        """Đọc file PDF"""
        try:
            content = []
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text.strip():
                        content.append(text)
            
            return "\n".join(content)
        except Exception as e:
            print(f"❌ Error reading PDF: {e}")
            return ""
    
    def _read_txt(self, file_path: Path) -> str:
        """Đọc file TXT"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"❌ Error reading TXT: {e}")
            return ""
    
    def _analyze_document_structure(self, content: str) -> Dict[str, Any]:
        """
        Phân tích cấu trúc document để xác định strategy chunking tối ưu
        """
        lines = content.split('\n')
        structure = {
            "total_lines": len(lines),
            "total_chars": len(content),
            "detected_language": self._detect_language(content),
            "headers": [],
            "qa_pairs": [],
            "lists": [],
            "chunking_strategy": "mixed"
        }
        
        # Phát hiện headers
        header_patterns = [
            r'^[A-Z][A-Z\s]+$',  # ALL CAPS
            r'^\d+\.\s+[A-Z]',   # 1. Title
            r'^\d+\.\d+\s+[A-Z]', # 1.1. Subtitle
            r'^[A-Z][^.!?]*:$',   # Title ending with :
            r'^[A-Z][^.!?]*\?$',  # Question format
        ]
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Kiểm tra header patterns
            for pattern in header_patterns:
                if re.match(pattern, line):
                    header_level = self._determine_header_level(line)
                    structure["headers"].append({
                        "text": line,
                        "line_number": i,
                        "level": header_level
                    })
                    break
            
            # Phát hiện Q&A pairs
            if '?' in line and len(line) < 200:  # Câu hỏi ngắn
                # Tìm câu trả lời ở dòng tiếp theo
                if i + 1 < len(lines) and lines[i + 1].strip():
                    answer = lines[i + 1].strip()
                    if len(answer) > 20:  # Câu trả lời có ý nghĩa
                        structure["qa_pairs"].append({
                            "question": line,
                            "answer": answer,
                            "line_number": i
                        })
            
            # Phát hiện lists
            if re.match(r'^[-•*]\s+', line) or re.match(r'^\d+\.\s+', line):
                structure["lists"].append({
                    "text": line,
                    "line_number": i
                })
        
        # Xác định strategy chunking tối ưu
        if len(structure["qa_pairs"]) > len(lines) * 0.1:  # >10% là Q&A
            structure["chunking_strategy"] = "qa_based"
        elif len(structure["headers"]) > len(lines) * 0.05:  # >5% là headers
            structure["chunking_strategy"] = "header_based"
        elif len(structure["lists"]) > len(lines) * 0.15:  # >15% là lists
            structure["chunking_strategy"] = "semantic_based"
        else:
            structure["chunking_strategy"] = "size_based"
        
        return structure
    
    def _determine_header_level(self, header_text: str) -> int:
        """Xác định cấp độ header"""
        if re.match(r'^\d+\.\d+\.\d+', header_text):
            return 3
        elif re.match(r'^\d+\.\d+', header_text):
            return 2
        elif re.match(r'^\d+\.', header_text):
            return 1
        elif header_text.isupper() and len(header_text) > 3:
            return 1
        else:
            return 0
    
    def _create_smart_chunks(self, content: str, file_path: Path, structure: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Tạo chunks thông minh dựa trên cấu trúc document
        """
        strategy = structure.get("chunking_strategy", "mixed")
        
        if strategy == "qa_based":
            return self._chunk_by_qa_pairs(content, structure, file_path)
        elif strategy == "header_based":
            return self._chunk_by_headers(content, structure, file_path)
        elif strategy == "semantic_based":
            return self._chunk_by_semantic_boundaries(content, structure, file_path)
        else:
            return self._chunk_by_size(content, file_path)
    
    def _chunk_by_headers(self, content: str, structure: Dict[str, Any], file_path: Path) -> List[DocumentChunk]:
        """Chunking dựa trên headers"""
        lines = content.split('\n')
        chunks = []
        chunk_index = 0
        
        headers = structure.get("headers", [])
        if not headers:
            return self._chunk_by_size(content, file_path)
        
        # Sắp xếp headers theo line number
        headers.sort(key=lambda x: x["line_number"])
        
        for i, header in enumerate(headers):
            start_line = header["line_number"]
            
            # Xác định end_line
            if i + 1 < len(headers):
                end_line = headers[i + 1]["line_number"]
            else:
                end_line = len(lines)
            
            # Tạo chunk content
            chunk_lines = lines[start_line:end_line]
            chunk_content = "\n".join(chunk_lines).strip()
            
            if len(chunk_content) > 50:  # Chỉ tạo chunk nếu có nội dung
                # Tính toán vị trí character
                start_char = len("\n".join(lines[:start_line])) + (1 if start_line > 0 else 0)
                end_char = len("\n".join(lines[:end_line]))
                
                # Tạo chunk
                chunk = DocumentChunk(
                    content=chunk_content,
                    metadata={
                        "header_text": header["text"],
                        "header_level": header["level"],
                        "chunk_strategy": "header_based"
                    },
                    chunk_id=self._generate_chunk_id(file_path, chunk_index),
                    source_file=file_path.name,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=end_char,
                    chunk_type="header",
                    chunk_strategy="header_based",
                    header_level=header["level"],
                    token_count=len(chunk_content.split())
                )
                chunks.append(chunk)
                chunk_index += 1
        
        return chunks
    
    def _chunk_by_qa_pairs(self, content: str, structure: Dict[str, Any], file_path: Path) -> List[DocumentChunk]:
        """Chunking dựa trên Q&A pairs"""
        qa_pairs = structure.get("qa_pairs", [])
        if not qa_pairs:
            return self._chunk_by_size(content, file_path)
        
        chunks = []
        chunk_index = 0
        
        for qa_pair in qa_pairs:
            question = qa_pair["question"]
            answer = qa_pair["answer"]
            
            # Tạo Q&A chunk
            qa_content = f"Q: {question}\nA: {answer}"
            
            # Tìm vị trí trong content
            start_pos = content.find(question)
            if start_pos != -1:
                end_pos = start_pos + len(qa_content)
                
                chunk = DocumentChunk(
                    content=qa_content,
                    metadata={
                        "question": question,
                        "answer": answer,
                        "chunk_strategy": "qa_based"
                    },
                    chunk_id=self._generate_chunk_id(file_path, chunk_index),
                    source_file=file_path.name,
                    chunk_index=chunk_index,
                    start_char=start_pos,
                    end_char=end_pos,
                    chunk_type="qa_pair",
                    chunk_strategy="qa_based",
                    qa_count=1,
                    token_count=len(qa_content.split())
                )
                chunks.append(chunk)
                chunk_index += 1
        
        return chunks
    
    def _chunk_by_size(self, content: str, file_path: Path) -> List[DocumentChunk]:
        """Chunking dựa trên kích thước cố định"""
        chunks = []
        chunk_index = 0
        start_pos = 0
        
        while start_pos < len(content):
            # Xác định end position
            end_pos = start_pos + self.chunk_size
            
            # Nếu không phải chunk cuối, tìm vị trí cắt phù hợp
            if end_pos < len(content):
                # Tìm vị trí cắt gần nhất (dấu chấm, xuống dòng)
                cut_positions = [
                    content.rfind('.', start_pos, end_pos),
                    content.rfind('\n', start_pos, end_pos),
                    content.rfind(' ', start_pos, end_pos)
                ]
                
                # Chọn vị trí cắt tốt nhất
                best_cut = max([pos for pos in cut_positions if pos > start_pos], default=end_pos)
                end_pos = best_cut + 1
            
            # Tạo chunk content
            chunk_content = content[start_pos:end_pos].strip()
            
            if chunk_content:
                chunk = DocumentChunk(
                    content=chunk_content,
                    metadata={
                        "chunk_strategy": "size_based",
                        "size": len(chunk_content)
                    },
                    chunk_id=self._generate_chunk_id(file_path, chunk_index),
                    source_file=file_path.name,
                    chunk_index=chunk_index,
                    start_char=start_pos,
                    end_char=end_pos,
                    chunk_type="content",
                    chunk_strategy="size_based",
                    token_count=len(chunk_content.split())
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Cập nhật start position cho chunk tiếp theo
            start_pos = end_pos
            
            # Thêm overlap
            if start_pos < len(content):
                start_pos = max(0, start_pos - self.chunk_overlap)
        
        return chunks
    
    def _chunk_by_semantic_boundaries(self, content: str, structure: Dict[str, Any], file_path: Path) -> List[DocumentChunk]:
        """Chunking dựa trên ranh giới ngữ nghĩa"""
        # Chia content thành các đoạn dựa trên dấu xuống dòng kép
        paragraphs = re.split(r'\n\s*\n', content)
        chunks = []
        chunk_index = 0
        current_chunk = ""
        start_pos = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Kiểm tra xem có thể thêm paragraph vào chunk hiện tại không
            if len(current_chunk) + len(paragraph) <= self.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                # Lưu chunk hiện tại
                if current_chunk:
                    end_pos = start_pos + len(current_chunk)
                    chunk = self._create_semantic_chunk(
                        current_chunk, file_path, chunk_index, start_pos, end_pos
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Bắt đầu chunk mới
                current_chunk = paragraph
                start_pos = content.find(paragraph, start_pos)
        
        # Lưu chunk cuối cùng
        if current_chunk:
            end_pos = start_pos + len(current_chunk)
            chunk = self._create_semantic_chunk(
                current_chunk, file_path, chunk_index, start_pos, end_pos
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_semantic_chunk(self, content: str, file_path: Path, chunk_index: int, start_pos: int, end_pos: int) -> DocumentChunk:
        """Tạo semantic chunk"""
        return DocumentChunk(
            content=content,
            metadata={
                "chunk_strategy": "semantic_based",
                "paragraph_count": content.count('\n\n') + 1
            },
            chunk_id=self._generate_chunk_id(file_path, chunk_index),
            source_file=file_path.name,
            chunk_index=chunk_index,
            start_char=start_pos,
            end_char=end_pos,
            chunk_type="content",
            chunk_strategy="semantic_based",
            token_count=len(content.split())
        )
    
    def _detect_language(self, text: str) -> str:
        """Phát hiện ngôn ngữ của text"""
        try:
            if LANGDETECT_AVAILABLE:
                # Lấy sample text để detect
                sample = text[:1000] if len(text) > 1000 else text
                lang = detect(sample)
                return lang
            else:
                # Fallback: kiểm tra các ký tự đặc biệt của tiếng Việt
                vietnamese_chars = set('àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ')
                text_chars = set(text.lower())
                
                if vietnamese_chars.intersection(text_chars):
                    return "vi"
                else:
                    return "en"
        except Exception:
            return "vi"  # Default to Vietnamese
    
    def _determine_document_type(self, filename: str) -> str:
        """Xác định loại document dựa trên tên file"""
        filename_lower = filename.lower()
        
        if "faq_behavior" in filename_lower:
            return "faq_behavior"
        elif "faq_training" in filename_lower:
            return "faq_training"
        elif "handbook" in filename_lower:
            return "handbook"
        elif "present" in filename_lower:
            return "presentation"
        else:
            return "unknown"
    
    def _generate_chunk_id(self, file_path: Path, chunk_index: int) -> str:
        """Tạo ID duy nhất cho chunk"""
        file_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
        return f"{file_hash}_{chunk_index}_{int(time.time())}"


class DocumentProcessorFactory:
    """Factory để tạo document processor"""
    
    @staticmethod
    def create_processor(chunk_size: int = 1000, chunk_overlap: int = 200) -> DocumentProcessor:
        """Tạo document processor với cấu hình cơ bản"""
        return DocumentProcessor(chunk_size, chunk_overlap)
    
    @staticmethod
    def create_processor_for_document_type(file_path: str) -> DocumentProcessor:
        """Tạo document processor tối ưu cho loại document cụ thể"""
        file_path = Path(file_path)
        filename_lower = file_path.name.lower()
        
        if "faq" in filename_lower:
            # FAQ documents: chunk nhỏ hơn, overlap ít hơn
            return DocumentProcessor(chunk_size=800, chunk_overlap=100)
        elif "handbook" in filename_lower:
            # Handbook: chunk lớn hơn để giữ context
            return DocumentProcessor(chunk_size=1200, chunk_overlap=300)
        elif "present" in filename_lower:
            # Presentation: chunk vừa phải
            return DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        else:
            # Default
            return DocumentProcessor()
    
    @staticmethod
    def create_trading_optimized_processor() -> DocumentProcessor:
        """Tạo document processor tối ưu cho trading documents"""
        return DocumentProcessor(
            chunk_size=1000,  # Kích thước phù hợp cho trading content
            chunk_overlap=200   # Overlap để giữ context
        )
