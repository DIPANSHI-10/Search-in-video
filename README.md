# Video Search App

## Overview
The Video Search App is a powerful tool that allows users to search for specific segments within video files based on transcribed audio content. By leveraging advanced audio processing, transcription, and embedding techniques, this application provides an intuitive interface for users to quickly find and play relevant video sections.

## Features
- **Audio Extraction**: Automatically extracts audio from video files for processing.
- **Transcription**: Converts audio to text using state-of-the-art speech recognition models.
- **Text Embeddings**: Generates embeddings for transcribed text to facilitate efficient searching.
- **Search Functionality**: Allows users to input queries and retrieve relevant video segments.
- **Video Playback**: Plays video segments at specified timestamps for seamless viewing.
- **Responsive GUI**: Features an impressive and user-friendly graphical interface for enhanced user experience.

## Project Structure
```
video-search-app
├── src
│   ├── app.py            # Main entry point for the application
│   ├── main.py           # Core logic for audio extraction, transcription, and search
│   ├── gui.py            # Graphical user interface design and event handling
│   ├── audio_utils.py    # Audio processing functions
│   ├── transcribe.py     # Audio transcription functionalities
│   ├── embeddings.py      # Functions for creating text embeddings
│   ├── search.py         # Search functionalities for querying transcribed text
│   └── video_player.py    # Video playback functions
├── requirements.txt      # Project dependencies
├── README.md             # Project documentation
└── .gitignore            # Files and directories to ignore in version control
```

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd video-search-app
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Run the application:
   ```
   python src/app.py
   ```

2. Follow the on-screen instructions to extract audio, transcribe it, and search for video segments.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.