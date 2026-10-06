import chromadb


# ==================================================
# CHROMADB CLIENT
# ==================================================

client = chromadb.PersistentClient(
    path="data/chroma_db"
)


# ==================================================
# COLLECTION
# ==================================================

collection = client.get_or_create_collection(
    name="persona_prep_documents"
)


# ==================================================
# STORE EMBEDDINGS
# ==================================================

def store_embeddings(
    chunks,
    embeddings,
    document_id,
    filename
):

    for i, (chunk, embedding) in enumerate(
        zip(chunks, embeddings)
    ):

        chunk_id = f"{document_id}_{i}"

        collection.upsert(
            ids=[chunk_id],

            documents=[chunk],

            embeddings=[
                embedding.tolist()
            ],

            metadatas=[
                {
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": i
                }
            ]
        )


# ==================================================
# SEARCH
# ==================================================

def search(
    query_embedding,
    top_k=3,
    document_id=None
):

    if document_id:

        results = collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],

            n_results=top_k,

            where={
                "document_id": document_id
            }
        )

    else:

        results = collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],

            n_results=top_k
        )

    return results


# ==================================================
# COLLECTION COUNT
# ==================================================

def get_collection_count():

    return collection.count()