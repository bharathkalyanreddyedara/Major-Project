-- ==========================================================
-- Supabase pgvector Schema for Agricultural Knowledge Base
-- Run this in your Supabase SQL Editor (https://supabase.com/dashboard)
-- ==========================================================

-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create Knowledge Documents Table
CREATE TABLE IF NOT EXISTS crop_knowledge_documents (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    content TEXT NOT NULL,
    source TEXT NOT NULL,              -- e.g. "crops/cotton.md"
    category TEXT NOT NULL,            -- e.g. "crops", "soils", "pest_and_disease"
    metadata JSONB DEFAULT '{}'::jsonb, -- e.g. {"crop": "Cotton", "soil": "Black"}
    embedding VECTOR(768),             -- 768 dimensions for Gemini text-embedding-004
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 3. Create HNSW High-Performance Vector Index
CREATE INDEX IF NOT EXISTS crop_knowledge_embedding_hnsw_idx 
ON crop_knowledge_documents 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 4. Create Match Documents Function for Vector Search
CREATE OR REPLACE FUNCTION match_crop_documents (
    query_embedding VECTOR(768),
    match_threshold FLOAT DEFAULT 0.25,
    match_count INT DEFAULT 5
)
RETURNS TABLE (
    id BIGINT,
    content TEXT,
    source TEXT,
    category TEXT,
    metadata JSONB,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        d.id,
        d.content,
        d.source,
        d.category,
        d.metadata,
        1 - (d.embedding <=> query_embedding) AS similarity
    FROM crop_knowledge_documents d
    WHERE 1 - (d.embedding <=> query_embedding) > match_threshold
    ORDER BY d.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
