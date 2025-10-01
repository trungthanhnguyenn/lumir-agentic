#!/usr/bin/env python3
"""
Ingest All Documents into RAG System with Deduplication
Process only new/changed documents, skip already processed ones
"""

import sys
from pathlib import Path
import time
import hashlib

# Add module to path
sys.path.append(str(Path(__file__).parent))

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
        # Check if any points exist from this file
        from qdrant_client.http import models as qdrant_models
        
        # Search for points with this source file
        filter_condition = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="source",
                    match=qdrant_models.MatchValue(value=str(file_path))
                )
            ]
        )
        
        # Scroll to find matching points
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

def ingest_documents():
    """Ingest all documents with deduplication"""
    print("🚀 Starting Document Ingestion into RAG System")
    print("=" * 80)
    
    try:
        from module.document.document_processor import DocumentProcessor
        from module.advanced_rag_orchestrator import AdvancedRAGOrchestrator
        
        # Initialize processors
        print("🔧 Initializing processors...")
        doc_processor = DocumentProcessor()
        orchestrator = AdvancedRAGOrchestrator(qdrant_host="localhost", qdrant_port=6333)
        
        # Create collection if not exists
        collection_name = "trading_docs"
        from module.database.qdrant_manager import CollectionConfig
        
        full_collection_name = f"{orchestrator.qdrant_manager.collection_prefix}_{collection_name}"
        if not orchestrator.qdrant_manager._collection_exists(full_collection_name):
            config = CollectionConfig(
                name=collection_name,
                vector_size=1024,
                distance="Cosine"
            )
            orchestrator.qdrant_manager.create_collection(config)
            print(f"✅ Created collection: {collection_name}")
        else:
            print(f"✅ Collection exists: {collection_name}")
        
        # Find all supported documents
        print("📂 Scanning for documents...")
        doc_extensions = ['.pdf', '.txt', '.md', '.docx']
        documents = []
        
        for ext in doc_extensions:
            documents.extend(Path('./trading_data').rglob(f'*{ext}'))
        
        if not documents:
            print("❌ No documents found in trading_data directory")
            return False
        
        print(f"📊 Found {len(documents)} documents")
        
        # Check current database status
        full_collection_name = f"{orchestrator.qdrant_manager.collection_prefix}_{collection_name}"
        try:
            collection_info = orchestrator.qdrant_manager.client.get_collection(full_collection_name)
            current_points = collection_info.points_count
            print(f"📋 Current database: {current_points} chunks already stored")
        except:
            current_points = 0
            print(f"📋 Current database: empty")
        
        print(f"\n🔄 Processing documents (only new/changed ones)...")
        
        processed_docs = 0
        skipped_docs = 0
        failed_docs = 0
        total_chunks = 0
        
        start_time = time.time()
        
        for i, doc_path in enumerate(documents):
            print(f"\n📄 [{i+1}/{len(documents)}] Checking: {doc_path.name}")
            
            try:
                # Check if document already exists
                if check_document_exists_in_qdrant(doc_path, orchestrator.qdrant_manager, collection_name):
                    print(f"  ⏭️  Already processed, skipping")
                    skipped_docs += 1
                    continue
                
                # Process document
                chunks, doc_info = doc_processor.process_document(doc_path)
                
                if not chunks:
                    print(f"  ⚠️  No chunks created, skipping")
                    failed_docs += 1
                    continue
                
                print(f"  📖 Document: {doc_path.name}")
                print(f"  📏 Type: {doc_info.document_type} → {len(chunks)} chunks")
                
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
                
                if stored_points > 0:
                    print(f"  ✅ Stored {stored_points}/{len(chunks)} chunks")
                    processed_docs += 1
                    total_chunks += stored_points
                    
                    # Show sample
                    if chunks:
                        sample = chunks[0].content[:100]
                        print(f"  📝 Sample: {sample}...")
                else:
                    print(f"  ❌ No chunks stored")
                    failed_docs += 1
                    
            except Exception as e:
                print(f"  ❌ Failed to process: {e}")
                failed_docs += 1
                continue
        
        processing_time = time.time() - start_time
        new_total_points = current_points + total_chunks
        
        print(f"\n" + "=" * 80)
        print(f"📊 INGESTION SUMMARY")
        print(f"=" * 80)
        print(f"📄 Total documents found: {len(documents)}")
        print(f"✅ Newly processed: {processed_docs}")
        print(f"⏭️  Already processed: {skipped_docs}")
        print(f"❌ Failed to process: {failed_docs}")
        print(f"🧩 New chunks stored: {total_chunks}")
        print(f"📋 Total chunks in database: {new_total_points}")
        print(f"⏱️  Processing time: {processing_time:.2f} seconds")
        
        if processed_docs > 0:
            print(f"\n🎉 DOCUMENT INGESTION COMPLETED!")
            print(f"🔍 System ready with {new_total_points} knowledge chunks")
            return True
        else:
            print(f"\n📋 ALL DOCUMENTS ALREADY UP TO DATE")
            return True
            
    except Exception as e:
        print(f"❌ Critical error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    success = ingest_documents()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()