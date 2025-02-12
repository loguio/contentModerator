import os
import time
import urllib.request
import json
import boto3
import cv2
from dotenv import load_dotenv
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

# Download NLTK data if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')


def check_filetype(filename):
    """
    Détermine le type de fichier en fonction de son extension.
    """
    file_basename = os.path.basename(filename)
    extension = file_basename.split(".")[-1]
    if extension in ["jpg", "png", "tiff", "svg"]:
        filetype = "image"
    elif extension in ["mp4", "avi", "mkv"]:
        filetype = "vidéo"
    else:
        filetype = None
    print(f"[INFO] : Le fichier {file_basename} est de type : {filetype}")
    return filetype


def extract_frame_video(video_path, frame_id):
    """
    Extrait une image spécifique d'une vidéo.
    """
    video = cv2.VideoCapture(video_path)
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, image = video.read()
    return image if ret else None


def get_aws_session(access_key,secret_key):
    """
    Crée et retourne une session AWS.
    """
    
    aws_session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="us-east-1"
    )
    return aws_session


def moderate_image(image_path, aws_service):
    """
    Détecte du contenu nécessitant une modération dans une image en utilisant Amazon Rekognition.
    """
    with open(image_path, 'rb') as image:
        response = aws_service.detect_moderation_labels(Image={'Bytes': image.read()})
    moderation_labels = [label['Name'] for label in response['ModerationLabels']]
    return moderation_labels


def get_text_from_speech(filename, aws_service, job_name, bucket_name):
    """
    Convertit de la parole en texte en utilisant AWS Transcribe.
    """
    s3 = boto3.client('s3',

aws_access_key_id=os.getenv("ACCESS_KEY"),
        aws_secret_access_key=os.getenv("SECRET_KEY"),
        region_name="us-east-1"


                      )
    s3.upload_file(filename, bucket_name, filename)
    aws_service.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': f's3://{bucket_name}/{filename}'},
        MediaFormat='mp4',
        LanguageCode='fr-FR'
    )
    while True:
        status = aws_service.get_transcription_job(TranscriptionJobName=job_name)
        if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            break
        time.sleep(5)
    if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
        response = urllib.request.urlopen(status['TranscriptionJob']['Transcript']['TranscriptFileUri'])
        data = json.loads(response.read())
        text = data['results']['transcripts'][0]['transcript']
        return text
    else:
        print(f"Transcription failed: {status['TranscriptionJob']['FailureReason']}")
        return None


def clean_text(raw_text):
    """
    Nettoie un texte en retirant les mots vides et en normalisant les mots en minuscules.
    """
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(raw_text.lower())
    stop_words = set(stopwords.words('french'))
    # Add more stop words if needed
    # stop_words.update(['word1', 'word2', ...])
    filtered_tokens = [w for w in tokens if w not in stop_words]
    cleaned_text = ' '.join(filtered_tokens)
    return cleaned_text


def extract_keyphrases(text, aws_service):
    """
    Extrait les expressions clés d'un texte et retourne les 10 expressions les plus pertinentes comme hashtags.
    """
    response = aws_service.detect_key_phrases(Text=text, LanguageCode='fr')
    keyphrases = [f"#{phrase['Text']}" for phrase in response['KeyPhrases'][:10]]
    return keyphrases


def detect_objects(image_path, aws_service):
    """
    Détecte les objets dans une image en utilisant Amazon Rekognition.
    """
    with open(image_path, 'rb') as image:
        response = aws_service.detect_labels(Image={'Bytes': image.read()}, MaxLabels=10, MinConfidence=50)
    objects = [label['Name'] for label in response['Labels']]
    return objects


def detect_celebrities(image_path, aws_service):
    """
    Identifie les célébrités dans une image en utilisant le service Amazon Rekognition.
    """
    with open(image_path, 'rb') as image:
        response = aws_service.recognize_celebrities(Image={'Bytes': image.read()})
    celebrities = [celebrity['Name'] for celebrity in response['CelebrityFaces'][:10]]
    return celebrities


def detect_emotions(image_path, aws_service):
    """
    Détecte les émotions sur les visages présents dans une image en utilisant Amazon Rekognition.
    """
    with open(image_path, 'rb') as image:
        response = aws_service.detect_faces(Image={'Bytes': image.read()}, Attributes=['ALL'])

    faces_info = []
    for faceDetail in response['FaceDetails']:
        print("[INFO] Visage détecté:")

        # Genre
        gender = faceDetail['Gender']['Value']
        gender_confidence = faceDetail['Gender']['Confidence']
        print(f"  - Genre: {gender} (confiance: {gender_confidence:.1f}%)")

        # Âge estimé
        age_low = faceDetail['AgeRange']['Low']
        age_high = faceDetail['AgeRange']['High']
        print(f"  - Âge estimé: {age_low}-{age_high} ans")

        # Émotions principales (3 émotions avec la plus haute confiance)
        emotions = sorted(faceDetail['Emotions'], key=lambda x: x['Confidence'], reverse=True)[:3]
        print("  - Émotions principales:")
        for emotion in emotions:
            print(f"    * {emotion['Type']}: {emotion['Confidence']:.1f}%")
        print("---")

        faces_info.append({
            'BoundingBox': faceDetail['BoundingBox'],
            'Emotions': emotions,
            'AgeRange': faceDetail['AgeRange'],
            'Gender': faceDetail['Gender']
        })

    return faces_info


def summarize_emotions(faces_info):
    """
    Résume les émotions détectées sur tous les visages d'une image.
    """
    total_faces = len(faces_info)
    if total_faces == 0:
        return {'total_faces': 0, 'dominant_emotion': None, 'emotion_stats': {}, 'age_stats': {}, 'gender_distribution': {}}

    emotion_counts = {}
    emotion_confidences = {}
    ages = []
    gender_counts = {'Male': 0, 'Female': 0}

    for face in faces_info:
        # Émotions (confiance > 50%)
        for emotion in face['Emotions']:
            if emotion['Confidence'] > 50:
                emotion_type = emotion['Type']
                emotion_counts[emotion_type] = emotion_counts.get(emotion_type, 0) + 1
                emotion_confidences[emotion_type] = emotion_confidences.get(emotion_type, 0) + emotion['Confidence']

        # Âge
        ages.append((face['AgeRange']['Low'] + face['AgeRange']['High']) / 2)

        # Genre
        gender_counts[face['Gender']['Value']] += 1

    # Émotion dominante (moyenne de confiance la plus élevée)
    dominant_emotion = max(emotion_confidences, key=lambda emotion: emotion_confidences[emotion] / emotion_counts[emotion], default=None)

    # Statistiques des émotions
    emotion_stats = {}
    for emotion_type in emotion_counts:
        emotion_stats[emotion_type] = {
            'count': emotion_counts[emotion_type],
            'avg_confidence': emotion_confidences[emotion_type] / emotion_counts[emotion_type]
        }

    # Statistiques d'âge
    age_stats = {
        'min': min(ages, default=None),
        'max': max(ages, default=None),
        'avg': sum(ages) / len(ages) if ages else None
    }

    # Distribution des genres
    gender_distribution = {
        'Male': gender_counts['Male'] / total_faces if total_faces else 0,
        'Female': gender_counts['Female'] / total_faces if total_faces else 0
    }

    summary = {
        'total_faces': total_faces,
        'dominant_emotion': dominant_emotion,
        'emotion_stats': emotion_stats,
        'age_stats': age_stats,
        'gender_distribution': gender_distribution
    }

    return summary


def process_media(media_file, rekognition, transcribe, comprehend, bucket_name):
    """
    Traite un fichier multimédia (image ou vidéo) pour modérer le contenu, détecter des objets/célébrités,
    transcrire le discours et extraire des expressions clés.
    """
    filetype = check_filetype(media_file)
    result = {}

    if filetype == "image":
        moderation_labels = moderate_image(media_file, rekognition)
        if moderation_labels:
            print(f"[INFO] : Contenu choquant détecté dans l'image : {moderation_labels}")
            result['moderation'] = moderation_labels  # Return None if inappropriate content is found
            return result
        else:
            objects = detect_objects(media_file, rekognition)
            celebrities = detect_celebrities(media_file, rekognition)
            emotions_info = detect_emotions(media_file, rekognition)
            emotion_summary = summarize_emotions(emotions_info)
            dominant_emotion = emotion_summary.get('dominant_emotion')
            hashtags = [f"#{obj}" for obj in objects] + [f"#{celeb}" for celeb in celebrities]
            if dominant_emotion:
                hashtags.append(f"#{dominant_emotion.lower()}")  # Add dominant emotion as hashtag
            result['hashtags'] = hashtags

    elif filetype == "vidéo":
        frame = extract_frame_video(media_file, 0)
        if frame is not None:
            temp_image_path = "temp_frame.jpg"
            cv2.imwrite(temp_image_path, frame)
            moderation_labels = moderate_image(temp_image_path, rekognition)
            os.remove(temp_image_path)  # Remove temporary image file

            if moderation_labels:
                print(f"[INFO] : Contenu choquant détecté dans la vidéo : {moderation_labels}")
                result['moderation'] = moderation_labels  # Return None if inappropriate content is found
                return result            
            else:
                job_name = f"transcribe-job-{time.time()}"
                text = get_text_from_speech(media_file, transcribe, job_name, bucket_name)
                if text:
                    cleaned_text = clean_text(text)
                    hashtags = extract_keyphrases(cleaned_text, comprehend)
                    print(hashtags)
                    result['subtitles'] = text
                    result['hashtags'] = hashtags
                else:
                    print("[INFO] : La transcription a échoué.")
                    return None
        else:
            print("[INFO] : Impossible d'extraire une image de la vidéo.")
            return None

    else:
        print("[INFO] : Type de fichier non pris en charge.")
        return None

    return result