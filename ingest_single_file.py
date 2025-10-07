#!/usr/bin/env python3
"""
Ingest Single Document into Existing Collection
"""

import sys
from pathlib import Path
import time
import hashlib
from typing import Optional

from module.document.document_processor_patch import create_patched_processor
from module.advanced_rag_orchestrator import AdvancedRAGOrchestrator
from qdrant_client.http import models as qdrant_models
from module.database.qdrant_manager import CollectionConfig

def get_file_hash(file_path: Path) -> str:
    """Generate hash for file to detect changes"""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return ""

def check_document_exists_in_qdrant(file_path: Path, qdrant, collection_name: str) -> bool:
    """Check if document already exists in Qdrant"""
    try:        
        filter_condition = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="source",
                    match=qdrant_models.MatchValue(value=str(file_path))
                )
            ]
        )
        
        points = qdrant.client.scroll(
            collection_name=collection_name,
            scroll_filter=filter_condition,
            limit=1,
            with_payload=False
        )[0]
        
        return len(points) > 0
        
    except Exception as e:
        print(f"    ⚠️  Error checking document existence: {e}")
        return False

def ingest_single_file(file_path: str, collection_name: str = "lumir_trading_data", force_update: bool = False) -> bool:
    """
    Ingest a single document into existing collection
    
    Args:
        file_path: Path to the specific document file
        collection_name: Target collection name (default: 'lumir_trading_data')
        force_update: Force reprocessing even if file exists (default: False)
    """
    print("🚀 Starting Single Document Ingestion")
    print("=" * 50)
    
    try:
        # Validate file exists
        doc_path = Path(file_path)
        if not doc_path.exists():
            print(f"❌ File not found: {file_path}")
            return False
        
        # Check file extension
        allowed_extensions = ['.pdf', '.txt', '.md', '.docx']
        if doc_path.suffix.lower() not in allowed_extensions:
            print(f"❌ Unsupported file type: {doc_path.suffix}")
            print(f"   Supported types: {', '.join(allowed_extensions)}")
            return False
        
        print(f"📄 Processing file: {doc_path.name}")
        print(f"📦 Target collection: {collection_name}")
        
        # Initialize processors
        print("🔧 Initializing processors...")
        doc_processor = create_patched_processor(chunk_size=1000, chunk_overlap=200)
        orchestrator = AdvancedRAGOrchestrator(qdrant_host="localhost", qdrant_port=1237)
        
        # Check collection exists
        if not orchestrator.qdrant_manager._collection_exists(collection_name):
            print(f"❌ Collection '{collection_name}' does not exist")
            print("   Please create the collection first or use ingest_documents.py")
            return False
        
        print(f"✅ Collection exists: {collection_name}")
        
        # Check if document already exists (unless force_update)
        if not force_update and check_document_exists_in_qdrant(doc_path, orchestrator.qdrant_manager, collection_name):
            print(f"⚠️  Document already exists in collection")
            print(f"   Use force_update=True to reprocess")
            return False
        
        # Check current database status
        try:
            collection_info = orchestrator.qdrant_manager.client.get_collection(collection_name)
            current_points = collection_info.points_count
            print(f"📋 Current database: {current_points} chunks")
        except:
            current_points = 0
            print(f"📋 Current database: empty")
        
        # Process document
        print(f"🔄 Processing document...")
        start_time = time.time()
        
        chunks, doc_info = doc_processor.process_document(doc_path)
        
        if not chunks:
            print(f"❌ No chunks created from document")
            return False
        
        print(f"📖 Document: {doc_path.name}")
        print(f"📏 Type: {doc_info.document_type} → {len(chunks)} chunks")
        
        # Store chunks in Qdrant
        stored_points = 0
        for chunk in chunks:
            try:
                embedding = orchestrator.embedding_manager.get_embedding(chunk.content)
                
                point_data = {
                    "id": f"{doc_path.stem}_chunk_{chunk.chunk_index}_{int(time.time()*1000)}",
                    "vector": embedding,
                    "payload": {
                        "content": chunk.content,
                        "source": str(doc_path),
                        "filename": doc_path.name,
                        "chunk_index": chunk.chunk_index,
                        "document_type": doc_info.document_type,
                        "file_hash": get_file_hash(doc_path),
                        "processed_time": time.time(),
                        "page_number": chunk.metadata.get("page_number"),
                        "section": chunk.metadata.get("section"),
                        "char_count": len(chunk.content)
                    }
                }
                
                success = orchestrator.qdrant_manager.upsert_points(collection_name, [point_data])
                if success:
                    stored_points += 1
                
            except Exception as e:
                print(f"    ❌ Failed to store chunk {chunk.chunk_index}: {e}")
        
        processing_time = time.time() - start_time
        new_total_points = current_points + stored_points
        
        print(f"\n" + "=" * 50)
        print(f"📊 INGESTION SUMMARY")
        print(f"=" * 50)
        print(f"📄 File: {doc_path.name}")
        print(f"🧩 Chunks stored: {stored_points}/{len(chunks)}")
        print(f"📋 Total chunks in collection: {new_total_points}")
        print(f"⏱️  Processing time: {processing_time:.2f} seconds")
        
        if stored_points > 0:
            print(f"\n🎉 DOCUMENT INGESTED SUCCESSFULLY!")
            if chunks:
                sample = chunks[0].content[:100]
                print(f"📝 Sample: {sample}...")
            return True
        else:
            print(f"\n❌ NO CHUNKS STORED")
            return False
            
    except Exception as e:
        print(f"❌ Critical error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function for command line usage"""
    if len(sys.argv) < 2:
        print("Usage: python ingest_single_file.py <file_path> [collection_name] [--force]")
        print("Example: python ingest_single_file.py 'TBI (Trader Behavioral Index) - Tài liệu RAG.docx' lumir_trading_data")
        print("Example: python ingest_single_file.py 'document.pdf' --force")
        sys.exit(1)
    
    file_path = sys.argv[1]
    collection_name = "lumir_trading_data"  # default
    force_update = False
    
    # Parse optional arguments
    for arg in sys.argv[2:]:
        if arg == "--force":
            force_update = True
        elif not arg.startswith("--"):
            collection_name = arg
    
    success = ingest_single_file(file_path, collection_name, force_update)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()