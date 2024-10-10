import os
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptAvailable, VideoUnavailable
from langdetect import detect
from googletrans import Translator
from tqdm import tqdm
import re
from dotenv import load_dotenv

load_dotenv()

# Set your YouTube Data API key and Channel ID
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
CHANNEL_ID = os.getenv("CHANNEL_ID")

# Create YouTube API service
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# Initialize Translator
translator = Translator()

# Get all video IDs from a YouTube channel
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

# Fetch transcripts for the given video IDs
def fetch_video_data(video_ids):
    video_data = {}
    # Create a progress bar
    pbar = tqdm(total=len(video_ids), desc="Processing videos", unit="video")
    
    for video_id in video_ids:
        try:
            # Fetch video details
            video_response = youtube.videos().list(
                part="snippet",
                id=video_id
            ).execute()

            video_title = video_response['items'][0]['snippet']['title']
            video_description = video_response['items'][0]['snippet']['description']
            
            # Clean the description to remove social handles and links
            cleaned_description = clean_description(video_description)

            # Attempt to fetch transcript using different methods
            transcript_text = None
            for method in range(1, 4):
                try:
                    if method == 1:
                        # Method 1: Try to get all transcripts
                        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                        transcript = transcript_list.find_transcript(['en'])
                        transcript_text = " ".join([entry['text'] for entry in transcript.fetch()])
                    elif method == 2:
                        # Method 2: Try to get transcript directly
                        transcript = YouTubeTranscriptApi.get_transcript(video_id)
                        transcript_text = " ".join([entry['text'] for entry in transcript])
                    else:
                        # Method 3: Try to get any available transcript and translate if necessary
                        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                        transcript = transcript_list.find_generated_transcript(['hi', 'en'])
                        transcript_text = " ".join([entry['text'] for entry in transcript.fetch()])
                    
                    if transcript_text:
                        break
                except Exception:
                    continue

            if transcript_text:
                # Detect language of the transcript
                detected_language = detect(transcript_text)
                if detected_language != 'en':  # If not in English, translate it to English
                    transcript_text = translator.translate(transcript_text, src=detected_language, dest='en').text

                video_data[video_id] = {
                    'title': video_title,
                    'description': cleaned_description,  # Use cleaned description
                    'transcript': transcript_text
                }

                # Create the 'transcriptions' folder if it doesn't exist
                os.makedirs('transcriptions', exist_ok=True)

                # Write the file in the 'transcriptions' folder
                with open(os.path.join('transcriptions', f"{video_id}_transcription.txt"), "w", encoding='utf-8') as f:
                    f.write(f"Title: {video_title}\n\nDescription: {cleaned_description}\n\nTranscript: {transcript_text}")
            else:
                pbar.write(f"Could not retrieve transcript for Video ID: {video_id}")

        except Exception as e:
            pbar.write(f"An error occurred while processing video {video_id}: {e}")
        
        finally:
            # Update the progress bar
            pbar.update(1)

    # Close the progress bar
    pbar.close()
    return video_data

def clean_description(description):
    # Remove URLs
    description = re.sub(r'http[s]?://\S+', '', description)
    # Remove social media handles (e.g., @username)
    description = re.sub(r'@\w+', '', description)
    # Remove any remaining unwanted characters (optional)
    description = re.sub(r'\s+', ' ', description).strip()
    return description

# Main Function
if __name__ == "__main__":
    # Step 1: Get all video IDs from the channel
    video_ids = get_all_video_ids(CHANNEL_ID)

    # Step 2: Fetch transcripts for the videos
    video_data = fetch_video_data(video_ids)

    # Optional: Print or process the video data
    # for video_id, data in video_data.items():
    #     print(f"\nData for Video ID {video_id}:")
    #     print(f"Title: {data['title']}")
    #     print(f"Description: {data['description']}")
    #     print(f"Transcript: {data['transcript']}")