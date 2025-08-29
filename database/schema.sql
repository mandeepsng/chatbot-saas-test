-- ChatFlow AI Vector Training Database Schema
-- PostgreSQL with pgvector extension

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "pgvector";
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ============================================================================
-- CORE ENTITIES
-- ============================================================================

-- Users table with authentication
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    name VARCHAR(255),
    company VARCHAR(255),
    plan_type VARCHAR(50) DEFAULT 'starter',
    last_login TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Add missing columns if table exists
ALTER TABLE users ADD COLUMN IF NOT EXISTS password_hash VARCHAR(255);
ALTER TABLE users ADD COLUMN IF NOT EXISTS company VARCHAR(255);
ALTER TABLE users ADD COLUMN IF NOT EXISTS last_login TIMESTAMP;

-- Chatbots table
CREATE TABLE IF NOT EXISTS chatbots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    industry VARCHAR(100),
    language VARCHAR(10) DEFAULT 'en',
    status VARCHAR(20) DEFAULT 'active', -- active, training, inactive
    model_version VARCHAR(50) DEFAULT 'gpt-4',
    training_data_count INTEGER DEFAULT 0,
    last_trained_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- TRAINING DATA STORAGE
-- ============================================================================

-- Raw training data uploaded by users
CREATE TABLE training_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chatbot_id UUID NOT NULL REFERENCES chatbots(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Content
    content_type VARCHAR(50) NOT NULL, -- 'conversation', 'faq', 'document', 'knowledge_base'
    source_file VARCHAR(255),
    raw_content TEXT NOT NULL,
    processed_content TEXT,
    
    -- Metadata
    title VARCHAR(500),
    category VARCHAR(100),
    tags TEXT[],
    language VARCHAR(10) DEFAULT 'en',
    
    -- Processing status
    status VARCHAR(20) DEFAULT 'pending', -- pending, processing, completed, failed
    processing_error TEXT,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Conversation pairs for Q&A training
CREATE TABLE conversation_pairs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chatbot_id UUID NOT NULL REFERENCES chatbots(id) ON DELETE CASCADE,
    training_data_id UUID REFERENCES training_data(id) ON DELETE CASCADE,
    
    -- Q&A Content
    user_message TEXT NOT NULL,
    bot_response TEXT NOT NULL,
    context TEXT, -- Previous conversation context
    
    -- Classification
    intent VARCHAR(100),
    confidence_score FLOAT DEFAULT 0.0,
    category VARCHAR(100),
    
    -- Quality metrics
    user_rating INTEGER, -- 1-5 stars
    admin_approved BOOLEAN DEFAULT false,
    quality_score FLOAT DEFAULT 0.0,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- VECTOR EMBEDDINGS STORAGE
-- ============================================================================

-- Vector embeddings for training data
CREATE TABLE training_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chatbot_id UUID NOT NULL REFERENCES chatbots(id) ON DELETE CASCADE,
    training_data_id UUID REFERENCES training_data(id) ON DELETE CASCADE,
    conversation_pair_id UUID REFERENCES conversation_pairs(id) ON DELETE CASCADE,
    
    -- Vector data
    embedding vector(1536), -- OpenAI text-embedding-3-large dimension
    embedding_model VARCHAR(100) DEFAULT 'text-embedding-3-large',
    embedding_version VARCHAR(20) DEFAULT 'v1',
    
    -- Source content
    source_text TEXT NOT NULL,
    source_type VARCHAR(50) NOT NULL, -- 'question', 'answer', 'document_chunk', 'knowledge'
    
    -- Metadata for retrieval
    chunk_index INTEGER DEFAULT 0,
    chunk_size INTEGER DEFAULT 0,
    overlap_tokens INTEGER DEFAULT 0,
    
    -- Performance metrics
    similarity_threshold FLOAT DEFAULT 0.8,
    usage_count INTEGER DEFAULT 0,
    last_used_at TIMESTAMP,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Context embeddings for conversation history
CREATE TABLE context_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chatbot_id UUID NOT NULL REFERENCES chatbots(id) ON DELETE CASCADE,
    conversation_id UUID, -- Reference to live conversations
    
    -- Vector data
    embedding vector(1536),
    context_window TEXT NOT NULL,
    context_type VARCHAR(50) DEFAULT 'conversation_history',
    
    -- Metadata
    message_count INTEGER DEFAULT 1,
    session_id VARCHAR(255),
    user_id UUID REFERENCES users(id),
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP DEFAULT (CURRENT_TIMESTAMP + INTERVAL '30 days')
);

-- ============================================================================
-- MODEL TRAINING TRACKING
-- ============================================================================

-- Training jobs and their status
CREATE TABLE training_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chatbot_id UUID NOT NULL REFERENCES chatbots(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Job details
    job_type VARCHAR(50) NOT NULL, -- 'initial_training', 'incremental', 'retraining'
    status VARCHAR(20) DEFAULT 'queued', -- queued, processing, completed, failed, cancelled
    
    -- Training configuration
    model_config JSONB DEFAULT '{}',
    training_params JSONB DEFAULT '{}',
    data_snapshot_id UUID,
    
    -- Progress tracking
    total_samples INTEGER DEFAULT 0,
    processed_samples INTEGER DEFAULT 0,
    progress_percentage FLOAT DEFAULT 0.0,
    
    -- Results
    model_metrics JSONB DEFAULT '{}',
    performance_scores JSONB DEFAULT '{}',
    error_message TEXT,
    
    -- Resource usage
    processing_time_seconds INTEGER,
    memory_usage_mb INTEGER,
    gpu_hours FLOAT DEFAULT 0,
    
    -- Timestamps
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model versions and deployments
CREATE TABLE model_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chatbot_id UUID NOT NULL REFERENCES chatbots(id) ON DELETE CASCADE,
    training_job_id UUID REFERENCES training_jobs(id),
    
    -- Version info
    version_number VARCHAR(20) NOT NULL,
    model_type VARCHAR(50) DEFAULT 'embedding_based',
    is_active BOOLEAN DEFAULT false,
    
    -- Model artifacts
    model_path VARCHAR(500),
    config_path VARCHAR(500),
    metrics_path VARCHAR(500),
    
    -- Performance metrics
    accuracy_score FLOAT,
    f1_score FLOAT,
    response_time_ms INTEGER,
    
    -- A/B testing
    traffic_percentage FLOAT DEFAULT 0.0,
    test_conversations INTEGER DEFAULT 0,
    test_satisfaction FLOAT,
    
    -- Timestamps
    deployed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- KNOWLEDGE BASE & RETRIEVAL
-- ============================================================================

-- Knowledge base entries with vector search
CREATE TABLE knowledge_base (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chatbot_id UUID NOT NULL REFERENCES chatbots(id) ON DELETE CASCADE,
    
    -- Content
    title VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,
    summary TEXT,
    
    -- Categorization
    category VARCHAR(100),
    tags TEXT[],
    keywords TEXT[],
    
    -- Vector for semantic search
    embedding vector(1536),
    
    -- Usage statistics
    access_count INTEGER DEFAULT 0,
    last_accessed_at TIMESTAMP,
    effectiveness_score FLOAT DEFAULT 0.5,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- FAQ entries with vector matching
CREATE TABLE faq_entries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chatbot_id UUID NOT NULL REFERENCES chatbots(id) ON DELETE CASCADE,
    
    -- Q&A Content
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    variations TEXT[], -- Question variations
    
    -- Vector embeddings
    question_embedding vector(1536),
    answer_embedding vector(1536),
    
    -- Metadata
    category VARCHAR(100),
    confidence_threshold FLOAT DEFAULT 0.8,
    priority INTEGER DEFAULT 0,
    
    -- Analytics
    match_count INTEGER DEFAULT 0,
    satisfaction_rating FLOAT,
    last_matched_at TIMESTAMP,
    
    -- Status
    is_active BOOLEAN DEFAULT true,
    reviewed_by_admin BOOLEAN DEFAULT false,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- ANALYTICS & MONITORING
-- ============================================================================

-- Training data quality metrics
CREATE TABLE data_quality_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chatbot_id UUID NOT NULL REFERENCES chatbots(id) ON DELETE CASCADE,
    training_data_id UUID REFERENCES training_data(id) ON DELETE CASCADE,
    
    -- Quality scores
    completeness_score FLOAT DEFAULT 0.0,
    consistency_score FLOAT DEFAULT 0.0,
    diversity_score FLOAT DEFAULT 0.0,
    relevance_score FLOAT DEFAULT 0.0,
    
    -- Analysis results
    duplicate_count INTEGER DEFAULT 0,
    outlier_count INTEGER DEFAULT 0,
    missing_fields TEXT[],
    
    -- Recommendations
    quality_issues JSONB DEFAULT '[]',
    improvement_suggestions TEXT[],
    
    -- Timestamps
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Vector similarity search logs
CREATE TABLE similarity_search_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chatbot_id UUID NOT NULL REFERENCES chatbots(id) ON DELETE CASCADE,
    
    -- Query details
    query_text TEXT NOT NULL,
    query_embedding vector(1536),
    search_type VARCHAR(50) DEFAULT 'similarity',
    
    -- Search parameters
    similarity_threshold FLOAT DEFAULT 0.8,
    max_results INTEGER DEFAULT 10,
    filters JSONB DEFAULT '{}',
    
    -- Results
    results_count INTEGER DEFAULT 0,
    top_similarity_score FLOAT,
    response_time_ms INTEGER,
    
    -- Context
    session_id VARCHAR(255),
    user_id UUID REFERENCES users(id),
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- PERFORMANCE OPTIMIZATION
-- ============================================================================

-- Indexes for vector similarity search (HNSW algorithm)
CREATE INDEX IF NOT EXISTS idx_training_embeddings_vector 
ON training_embeddings USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_context_embeddings_vector 
ON context_embeddings USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_knowledge_base_vector 
ON knowledge_base USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_faq_question_vector 
ON faq_entries USING hnsw (question_embedding vector_cosine_ops);

-- Traditional indexes for filtering
CREATE INDEX IF NOT EXISTS idx_training_embeddings_chatbot 
ON training_embeddings(chatbot_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_conversation_pairs_chatbot 
ON conversation_pairs(chatbot_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_training_data_status 
ON training_data(status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_training_jobs_status 
ON training_jobs(status, created_at DESC);

-- Full text search indexes
CREATE INDEX IF NOT EXISTS idx_training_data_content_fts 
ON training_data USING gin(to_tsvector('english', processed_content));

CREATE INDEX IF NOT EXISTS idx_knowledge_base_content_fts 
ON knowledge_base USING gin(to_tsvector('english', content));

-- ============================================================================
-- ROW LEVEL SECURITY (RLS)
-- ============================================================================

-- Enable RLS on all tenant tables
ALTER TABLE chatbots ENABLE ROW LEVEL SECURITY;
ALTER TABLE training_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversation_pairs ENABLE ROW LEVEL SECURITY;
ALTER TABLE training_embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE training_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledge_base ENABLE ROW LEVEL SECURITY;
ALTER TABLE faq_entries ENABLE ROW LEVEL SECURITY;

-- RLS Policies (users can only access their own data)
CREATE POLICY chatbot_isolation ON chatbots
    FOR ALL USING (user_id = current_setting('app.user_id')::uuid);

CREATE POLICY training_data_isolation ON training_data
    FOR ALL USING (user_id = current_setting('app.user_id')::uuid);

CREATE POLICY training_embeddings_isolation ON training_embeddings
    FOR ALL USING (chatbot_id IN (
        SELECT id FROM chatbots WHERE user_id = current_setting('app.user_id')::uuid
    ));

-- ============================================================================
-- UTILITY FUNCTIONS
-- ============================================================================

-- Function to calculate vector similarity
CREATE OR REPLACE FUNCTION calculate_similarity(
    vec1 vector,
    vec2 vector
) RETURNS float AS $$
BEGIN
    RETURN 1 - (vec1 <=> vec2);
END;
$$ LANGUAGE plpgsql;

-- Function to find similar embeddings
CREATE OR REPLACE FUNCTION find_similar_embeddings(
    chatbot_uuid UUID,
    query_vector vector,
    similarity_threshold float DEFAULT 0.8,
    max_results integer DEFAULT 10
) RETURNS TABLE(
    id UUID,
    similarity_score float,
    source_text text,
    source_type text
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        te.id,
        (1 - (te.embedding <=> query_vector))::float as similarity_score,
        te.source_text,
        te.source_type
    FROM training_embeddings te
    WHERE te.chatbot_id = chatbot_uuid
    AND (1 - (te.embedding <=> query_vector)) >= similarity_threshold
    ORDER BY te.embedding <=> query_vector
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

-- Function to update chatbot training stats
CREATE OR REPLACE FUNCTION update_chatbot_training_stats(chatbot_uuid UUID)
RETURNS void AS $$
BEGIN
    UPDATE chatbots SET
        training_data_count = (
            SELECT COUNT(*) FROM training_data 
            WHERE chatbot_id = chatbot_uuid AND status = 'completed'
        ),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = chatbot_uuid;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- TRIGGERS
-- ============================================================================

-- Trigger to update training stats when training data changes
CREATE OR REPLACE FUNCTION trigger_update_training_stats()
RETURNS trigger AS $$
BEGIN
    PERFORM update_chatbot_training_stats(COALESCE(NEW.chatbot_id, OLD.chatbot_id));
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER training_data_stats_trigger
    AFTER INSERT OR UPDATE OR DELETE ON training_data
    FOR EACH ROW
    EXECUTE FUNCTION trigger_update_training_stats();

-- Trigger to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS trigger AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply update timestamp trigger to all relevant tables
CREATE TRIGGER update_chatbots_updated_at
    BEFORE UPDATE ON chatbots
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_training_data_updated_at
    BEFORE UPDATE ON training_data
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_conversation_pairs_updated_at
    BEFORE UPDATE ON conversation_pairs
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- ============================================================================
-- SAMPLE DATA FOR TESTING
-- ============================================================================

-- Insert sample user (for testing)
INSERT INTO users (id, email, name, plan_type) 
VALUES (
    '123e4567-e89b-12d3-a456-426614174000',
    'demo@chatflowai.com',
    'Demo User',
    'professional'
) ON CONFLICT (email) DO NOTHING;

-- Insert sample chatbot
INSERT INTO chatbots (id, user_id, name, description, industry) 
VALUES (
    '223e4567-e89b-12d3-a456-426614174000',
    '123e4567-e89b-12d3-a456-426614174000',
    'Customer Support Bot',
    'Handles customer inquiries and support tickets',
    'SaaS'
) ON CONFLICT (id) DO NOTHING;

-- Performance monitoring views
CREATE OR REPLACE VIEW training_performance_summary AS
SELECT 
    c.name as chatbot_name,
    c.user_id,
    COUNT(td.id) as total_training_data,
    COUNT(te.id) as total_embeddings,
    AVG(te.usage_count) as avg_embedding_usage,
    MAX(td.created_at) as last_data_upload,
    MAX(tj.completed_at) as last_training_completion
FROM chatbots c
LEFT JOIN training_data td ON c.id = td.chatbot_id AND td.status = 'completed'
LEFT JOIN training_embeddings te ON c.id = te.chatbot_id
LEFT JOIN training_jobs tj ON c.id = tj.chatbot_id AND tj.status = 'completed'
GROUP BY c.id, c.name, c.user_id;

COMMENT ON TABLE training_data IS 'Stores raw training data uploaded by users for chatbot training';
COMMENT ON TABLE training_embeddings IS 'Vector embeddings generated from training data for similarity search';
COMMENT ON TABLE conversation_pairs IS 'Question-answer pairs extracted from training data';
COMMENT ON TABLE training_jobs IS 'Tracks model training job status and progress';
COMMENT ON TABLE knowledge_base IS 'Structured knowledge base entries with vector search capabilities';
COMMENT ON TABLE faq_entries IS 'FAQ entries with vector-based matching for quick responses';