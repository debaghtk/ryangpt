import os
import re
import time
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptAvailable, VideoUnavailable
from langdetect import detect
from tqdm import tqdm
from dotenv import load_dotenv
# Use deep-translator instead of googletrans
from deep_translator import GoogleTranslator

load_dotenv()

# Set your YouTube Data API key and Channel ID
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
CHANNEL_ID = os.getenv("CHANNEL_ID")

# Create YouTube API service
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# Initialize Translator
translator = GoogleTranslator(source='auto', target='en')

def get_all_video_ids(channel_id):
    video_ids = []
    next_page_token = None

    while True:
        request = youtube.search().list(
            part="id",
            channelId=channel_id,
            maxResults=50,
            pageToken=next_page_token,
            type="video"
        )
        response = request.execute()

        for item in response["items"]:
            if "videoId" in item["id"]:
                video_ids.append(item["id"]["videoId"])

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return video_ids

def clean_description(description):
    # Remove URLs
    description = re.sub(r'http[s]?://\S+', '', description)
    # Remove social media handles (e.g., @username)
    description = re.sub(r'@\w+', '', description)
    # Remove any remaining unwanted characters (optional)
    description = re.sub(r'\s+', ' ', description).strip()
    return description

# Fetch transcripts for the given video IDs
def fetch_video_data(video_ids):
    video_data = {}
    failed_videos = []  # List to store video IDs for which transcription failed

    # Create a progress bar
    pbar = tqdm(total=len(video_ids), desc="Processing videos", unit="video")
    
    for video_id in video_ids:
        # Check if transcription file already exists
        transcription_file = os.path.join('transcriptions', f"{video_id}_transcription.txt")
        if os.path.exists(transcription_file):
            pbar.write(f"Skipping Video ID: {video_id} (transcription already exists)")
            pbar.update(1)
            continue  # Skip to the next video

        try:
            # Fetch video details
            video_response = youtube.videos().list(
                part="snippet",
                id=video_id
            ).execute()

            if not video_response['items']:
                pbar.write(f"No video details found for Video ID: {video_id}")
                failed_videos.append(video_id)
                continue  # Skip to the next video

            video_item = video_response['items'][0]
            snippet = video_item.get('snippet', {})

            video_title = snippet.get('title', '')
            video_description = snippet.get('description', '')

            # Clean the description to remove social handles and links
            cleaned_description = clean_description(video_description)

            # Attempt to fetch transcript using different methods
            transcript_entries = None
            for method in range(1, 4):
                try:
                    if method == 1:
                        # Method 1: Try to get all transcripts
                        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                        transcript = transcript_list.find_transcript(['en'])
                        transcript_entries = transcript.fetch()
                    elif method == 2:
                        # Method 2: Try to get transcript directly
                        transcript_entries = YouTubeTranscriptApi.get_transcript(video_id)
                    else:
                        # Method 3: Try to get any available transcript and translate if necessary
                        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                        transcript = transcript_list.find_generated_transcript(['hi', 'en'])
                        transcript_entries = transcript.fetch()
                    
                    if transcript_entries:
                        break
                except Exception:
                    continue

            if transcript_entries is not None and len(transcript_entries) > 0:
                # Detect language of the transcript
                transcript_text = " ".join([entry['text'] for entry in transcript_entries if entry.get('text')])
                detected_language = detect(transcript_text)

                if detected_language != 'en':
                    # Translate each entry individually
                    for entry in transcript_entries:
                        text_to_translate = entry.get('text', '')
                        if text_to_translate:
                            try:
                                # Use deep-translator
                                translated_text = translator.translate(text_to_translate)
                                entry['text'] = translated_text
                            except Exception as e:
                                print(f"Translation failed for video {video_id} at entry starting at {entry.get('start')}: {e}")
                                import traceback
                                traceback.print_exc()
                                entry['text'] = text_to_translate  # Fallback to original text
                            # Include a delay to prevent rate limiting
                            time.sleep(0.5)
                        else:
                            entry['text'] = ''

                # Create the 'transcriptions' folder if it doesn't exist
                os.makedirs('transcriptions', exist_ok=True)

                # Write the file in the 'transcriptions' folder with timestamps
                transcription_file = os.path.join('transcriptions', f"{video_id}_transcription.txt")
                with open(transcription_file, "w", encoding='utf-8') as f:
                    f.write(f"Title: {video_title}\n\nDescription: {cleaned_description}\n\nTranscript:\n")
                    for entry in transcript_entries:
                        start_time = entry.get('start')
                        duration = entry.get('duration')
                        text = entry.get('text')
                        if start_time is not None and duration is not None and text is not None:
                            f.write(f"[{start_time:.2f} - {start_time + duration:.2f}] {text}\n")

                # Update video data
                video_data[video_id] = {
                    'title': video_title,
                    'description': cleaned_description,
                    'transcript_entries': transcript_entries
                }

            else:
                pbar.write(f"Could not retrieve transcript for Video ID: {video_id}")
                failed_videos.append(video_id)  # Log the failed video ID

        except Exception as e:
            pbar.write(f"An error occurred while processing video {video_id}: {e}")
            import traceback
            pbar.write(traceback.format_exc())  # Output the traceback
            failed_videos.append(video_id)  # Log the failed video ID

        finally:
            # Update the progress bar
            pbar.update(1)

    # Close the progress bar
    pbar.close()

    # Write failed video IDs to a separate file
    if failed_videos:
        with open('failed_videos.txt', 'w') as fv:
            for vid in failed_videos:
                fv.write(f"{vid}\n")
        print(f"\nLogged failed video IDs to 'failed_videos.txt'.")

    return video_data

# Main Function
if __name__ == "__main__":
    # Step 1: Get all video IDs from the channel
    video_ids = get_all_video_ids(CHANNEL_ID)

    # Step 2: Fetch transcripts for the videos
    video_data = fetch_video_data(video_ids)