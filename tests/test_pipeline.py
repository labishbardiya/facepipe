
def test_pipeline_initialization():
    from facepipe.core.pipeline import RecognitionPipeline
    pipeline = RecognitionPipeline()
    assert not pipeline._is_initialized

    # We do a dry run init without models
    pipeline.initialize()
    assert pipeline._is_initialized
