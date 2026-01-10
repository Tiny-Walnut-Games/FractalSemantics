# FractalStat Hugging Face Space - Deployment Guide

## Quick Deployment

1. **Prerequisites Check**
   ```bash
   python setup_hf_space.py
   ```

2. **Install Dependencies** (if needed)
   ```bash
   pip install -r requirements_hf.txt
   ```

3. **Create Hugging Face Space**
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose "Gradio" as the Space type
   - Upload all project files

4. **Configure Environment**
   - Set `PYTHONUNBUFFERED=1`
   - Set `GRADIO_SERVER_NAME=0.0.0.0`
   - Set `GRADIO_SERVER_PORT=7860`

5. **Deploy**
   - Hugging Face will automatically build and deploy
   - Monitor the build logs for any issues

## File Structure

Required files for deployment:
```
fractalstat-hf-space/
├── app.py                    # Main application
├── requirements_hf.txt       # Dependencies
├── app.yaml                  # Hugging Face configuration
├── README_HF_SPACE.md        # Documentation
├── setup_hf_space.py         # Setup utility
└── fractalstat/              # Experiment modules
    ├── __init__.py
    ├── fractalstat_experiments.py
    ├── fractalstat_entity.py
    └── config/
        └── __init__.py
```

## Troubleshooting

### Import Errors
- Ensure all FractalStat modules are uploaded
- Check Python version compatibility (3.8+)
- Verify dependencies are installed

### Memory Issues
- Increase memory allocation in app.yaml
- Monitor experiment memory usage
- Consider running experiments sequentially

### Performance Issues
- Enable caching for repeated experiments
- Optimize visualization rendering
- Monitor CPU and memory usage

## Support

For issues and questions:
- Check the troubleshooting section above
- Review the main README_HF_SPACE.md
- Report bugs via GitHub issues
