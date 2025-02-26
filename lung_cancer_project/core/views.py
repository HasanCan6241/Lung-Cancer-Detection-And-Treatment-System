from django.shortcuts import render, redirect,get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import UserRegistrationForm, ContactForm, PredictionForm
from .models import FAQ, Prediction,ChatMessage
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
from django.http import HttpResponse
import csv
import os
import zipfile
from zipfile import ZipFile
from io import BytesIO
from django.db.models import Avg, Count
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import google.generativeai as genai
from django.conf import settings
from typing import Dict
from django.http import JsonResponse
from django.views.generic import TemplateView
import logging
from typing import Dict, Any, Optional
from functools import lru_cache
from django.utils.decorators import method_decorator
from django.contrib.auth.mixins import LoginRequiredMixin


def home(request):
    return render(request, 'home.html')


def about(request):
    return render(request, 'about.html')

@login_required
def analysis_dashboard(request):
    predictions = Prediction.objects.all()

    # Temel istatistikler
    # Base statistics
    context = {
        'total_predictions': predictions.count(),
        'positive_rate': round(predictions.filter(prediction_result='Normal').count() / predictions.count() * 100,
                               2) if predictions.exists() else 0,

        'avg_confidence': round(predictions.aggregate(Avg('confidence_score'))['confidence_score__avg'] or 0, 2),
        'avg_age': round(predictions.aggregate(Avg('age'))['age__avg'] or 0, 1),
    }

    # Update color mapping for four categories
    color_map = {
        'Adenocarcinoma': '#FF6B6B',
        'Large Cell Carcinoma': '#4ECDC4',
        'Normal': '#45B7D1',
        'Squamous Cell Carcinoma': '#96CEB4'
    }

    # 1. Tahmin Sonucu vs Cinsiyet
    df_gender = pd.DataFrame(list(predictions.values('gender', 'prediction_result')))
    fig_gender = px.histogram(
        df_gender,
        x='gender',
        color='prediction_result',
        barmode='group',
        title='Cinsiyet ve Tahmin Sonuçları',
        labels={'gender': 'Cinsiyet', 'count': 'Sayı', 'prediction_result': 'Tahmin Sonucu'},
        color_discrete_map=color_map
    )
    context['gender_prediction_plot'] = fig_gender.to_html(full_html=False)

    # 2. Tahmin Sonucu vs Sigara İçme Geçmişi
    df_smoking = pd.DataFrame(list(predictions.values('smoking_history', 'prediction_result')))
    fig_smoking = px.histogram(
        df_smoking,
        x='smoking_history',
        color='prediction_result',
        barmode='group',
        title='Sigara İçme Geçmişi ve Tahmin Sonuçları',
        labels={'smoking_history': 'Sigara İçme Geçmişi', 'count': 'Sayı'},
        color_discrete_map=color_map
    )

    context['smoking_prediction_plot'] = fig_smoking.to_html(full_html=False)

    # 3. Tahmin Sonucu vs Mesleki Maruziyet
    df_exposure = pd.DataFrame(list(predictions.values('occupational_exposure', 'prediction_result')))
    fig_exposure = px.histogram(
        df_exposure,
        x='occupational_exposure',
        color='prediction_result',
        barmode='group',
        title='Mesleki Maruziyet ve Tahmin Sonuçları',
        labels={'occupational_exposure': 'Mesleki Maruziyet'},
        color_discrete_map=color_map
    )
    context['occupational_exposure_plot'] = fig_exposure.to_html(full_html=False)

    # 4. Sigara İçme Yılları ve Tahmin Sonucu
    df_years = pd.DataFrame(list(predictions.values('smoking_years', 'prediction_result')))
    df_years = df_years.dropna()  # NaN değerleri temizle
    fig_years = px.box(
        df_years,
        x='prediction_result',
        y='smoking_years',
        title='Sigara İçme Yılları ve Tahmin Sonucu',
        color='prediction_result',
        color_discrete_map=color_map
    )
    context['smoking_years_plot'] = fig_years.to_html(full_html=False)

    # 5. Günlük Sigara Miktarı ve Tahmin Sonucu
    df_daily = pd.DataFrame(list(predictions.values('cigarettes_per_day', 'prediction_result')))
    fig_daily = px.box(
        df_daily,
        x='prediction_result',
        y='cigarettes_per_day',
        title='Günlük Sigara Miktarı ve Tahmin Sonucu',
        color='prediction_result',
        color_discrete_map=color_map
    )
    context['daily_cigarettes_plot'] = fig_daily.to_html(full_html=False)

    # 6. Tahmin Sonucu vs Semptomlar
    symptoms_data = {
        'Semptom': ['Nefes Darlığı', 'Göğüs Ağrısı', 'Kilo Kaybı', 'Yorgunluk', 'Balgam Üretimi'],
    }

    for cancer_type in color_map.keys():
        symptoms_data[cancer_type] = [
            predictions.filter(shortness_of_breath=True, prediction_result=cancer_type).count(),
            predictions.filter(chest_pain=True, prediction_result=cancer_type).count(),
            predictions.filter(weight_loss=True, prediction_result=cancer_type).count(),
            predictions.filter(fatigue=True, prediction_result=cancer_type).count(),
            predictions.filter(sputum_production=True, prediction_result=cancer_type).count(),
        ]

    # Update symptoms plot
    fig_symptoms = go.Figure()
    for cancer_type in color_map.keys():
        fig_symptoms.add_trace(go.Bar(
            name=cancer_type,
            x=symptoms_data['Semptom'],
            y=symptoms_data[cancer_type],
            marker_color=color_map[cancer_type]
        ))

    fig_symptoms.update_layout(barmode='group', title='Symptom Distribution and Prediction Results')
    context['symptoms_prediction_plot'] = fig_symptoms.to_html(full_html=False)

    # 7. Güven Skoru ve Tahmin Sonucu
    df_confidence = pd.DataFrame(list(predictions.values('confidence_score', 'prediction_result')))
    fig_confidence = px.box(
        df_confidence,
        x='prediction_result',
        y='confidence_score',
        title='Güven Skoru Dağılımı',
        color='prediction_result',
        color_discrete_map=color_map
    )
    context['confidence_plot'] = fig_confidence.to_html(full_html=False)

    # 8. Yaş Dağılımı
    df_age = pd.DataFrame(list(predictions.values('age', 'prediction_result')))
    fig_age = px.histogram(
        df_age,
        x='age',
        nbins=30,
        title='Yaş Dağılımı',
        color='prediction_result',
        color_discrete_map=color_map
    )
    context['age_distribution_plot'] = fig_age.to_html(full_html=False)

    # 9. Sigara İçme Süresi ve Günlük Sigara Kullanımı Dağılımı
    df_smoking_pattern = pd.DataFrame(
        list(predictions.values('smoking_years', 'cigarettes_per_day', 'prediction_result')))
    fig_smoking_pattern = px.scatter(
        df_smoking_pattern,
        x='smoking_years',
        y='cigarettes_per_day',
        color='prediction_result',
        title='Sigara İçme Süresi ve Günlük Kullanım İlişkisi',
        labels={'smoking_years': 'Sigara İçme Yılları', 'cigarettes_per_day': 'Günlük Sigara Sayısı'},
        color_discrete_map=color_map
    )
    context['smoking_distribution_plot'] = fig_smoking_pattern.to_html(full_html=False)

    # 10. Öksürük Süresi Dağılımı
    df_cough = pd.DataFrame(list(predictions.values('chronic_cough', 'prediction_result')))
    fig_cough = px.histogram(
        df_cough,
        x='chronic_cough',
        color='prediction_result',
        barmode='group',
        title='Kronik Öksürük ve Tahmin Sonuçları',
        labels={'chronic_cough': 'Kronik Öksürük', 'count': 'Sayı'},
        color_discrete_map=color_map
    )
    context['cough_prediction_plot'] = fig_cough.to_html(full_html=False)

    # 11. Cinsiyet Dağılımı
    df_gender_ratio = pd.DataFrame(list(predictions.values('gender')))
    fig_gender_ratio = px.pie(
        df_gender_ratio,
        names='gender',
        title='Cinsiyet Dağılımı',
        color_discrete_sequence=['#FF6B6B', '#4ECDC4']
    )
    context['gender_ratio_plot'] = fig_gender_ratio.to_html(full_html=False)

    df_location = pd.DataFrame(list(predictions.values('location')))

    # Lokasyonlara göre dağılımı gruplandır
    df_location['location'] = df_location['location'].str.upper()
    df_location_count = df_location['location'].value_counts().reset_index()
    df_location_count.columns = ['location', 'count']

    # Çubuk grafik oluştur
    fig_location = px.bar(
        df_location_count,
        x='location',
        y='count',
        title='Lokasyon Dağılımı',
        labels={'location': 'Lokasyon', 'count': 'Sayı'},
        color='location'
    )

    context['location_distribution_plot'] = fig_location.to_html(full_html=False)


    # 12. Sigara Kullanıcı Oranı
    smoking_status = {
        'Durum': ['Sigara Kullanıyor', 'Kullanmıyor'],
        'Sayı': [
            predictions.filter(smoking_history=True).count(),
            predictions.filter(smoking_history=False).count()
        ]
    }
    fig_smoking_ratio = px.pie(
        smoking_status,
        values='Sayı',
        names='Durum',
        title='Sigara Kullanım Oranı',
        color_discrete_sequence=['#FF6B6B', '#4ECDC4']
    )
    context['smoking_ratio_plot'] = fig_smoking_ratio.to_html(full_html=False)

    # 13. Mesleki Maruziyet Oranı
    exposure_status = {
        'Durum': ['Maruziyet Var', 'Maruziyet Yok'],
        'Sayı': [
            predictions.filter(occupational_exposure=True).count(),
            predictions.filter(occupational_exposure=False).count()
        ]
    }
    fig_exposure_ratio = px.pie(
        exposure_status,
        values='Sayı',
        names='Durum',
        title='Mesleki Maruziyet Oranı',
        color_discrete_sequence=['#FF6B6B', '#4ECDC4']
    )
    context['exposure_ratio_plot'] = fig_exposure_ratio.to_html(full_html=False)

    # 14. Tahmin Sonuçlarının Tarihsel Değişimi
    df_time = pd.DataFrame(list(predictions.values('created_at', 'prediction_result')))
    df_time['created_at'] = pd.to_datetime(df_time['created_at'])
    df_time = df_time.set_index('created_at')
    df_time_grouped = df_time.groupby([pd.Grouper(freq='M'), 'prediction_result']).size().unstack(fill_value=0)

    fig_time = go.Figure()

    color_map = {
        'Adenocarcinoma': '#FF6B6B',
        'Large Cell Carcinoma': '#4ECDC4',
        'Normal': '#45B7D1',
        'Squamous Cell Carcinoma': '#96CEB4'
    }

    for cancer_type in color_map:
        fig_time.add_trace(go.Scatter(
            x=df_time_grouped.index,
            y=df_time_grouped[cancer_type],
            name=cancer_type,
            line=dict(color=color_map[cancer_type])
        ))

    fig_time.update_layout(
        title='Prediction Results Over Time',
        xaxis_title='Date',
        yaxis_title='Number of Predictions'
    )
    context['time_series_plot'] = fig_time.to_html(full_html=False)

    # 3. Yaş Dağılımı ve Tahmin Sonuçları
    df_age = pd.DataFrame(list(predictions.values('age', 'prediction_result')))
    fig_age = px.histogram(
        df_age,
        x='age',
        color='prediction_result',
        nbins=20,  # Histogram için yaş aralıklarını belirler
        title='Yaş Dağılımı ve Tahmin Sonuçları',
        labels={'age': 'Yaş', 'count': 'Sayı'},
        color_discrete_map=color_map
    )
    context['age_prediction_plot'] = fig_age.to_html(full_html=False)

    # 8. Lokasyon Bazlı Dağılım
    df_location = pd.DataFrame(list(predictions.values('location', 'prediction_result')))

    df_location['location'] = df_location['location'].str.upper()

    fig_location = px.histogram(
        df_location,
        x='location',
        color='prediction_result',
        barmode='group',
        title='Lokasyon Bazlı Tahmin Dağılımı',
        labels={'location': 'Lokasyon', 'count': 'Sayı'},
        color_discrete_map=color_map
    )
    context['location_plot'] = fig_location.to_html(full_html=False)

    return render(request, 'users-panel/analysis_dashboard.html', context)


def faq(request):
    faqs = FAQ.objects.all().order_by('-created_at')
    return render(request, 'faq.html', {'faqs': faqs})


def contact(request):
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Mesajınız başarıyla gönderildi!')
            return redirect('contact')
    else:
        form = ContactForm()
    return render(request, 'contact.html', {'form': form})


def register(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Hesabınız başarıyla oluşturuldu!')
            return redirect('login')
    else:
        form = UserRegistrationForm()
    return render(request, 'registration/register.html', {'form': form})


class ResNetLungCancer(nn.Module):
    def __init__(self, num_classes, use_pretrained=True):
        super(ResNetLungCancer, self).__init__()
        if use_pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None

        self.resnet = resnet50(weights=weights)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.resnet(x)
        return self.fc(x)

@login_required
def dashboard(request):
    prediction_result = None  # Bu, tahmin sonucu verisini tutacak

    if request.method == 'POST':
        form = PredictionForm(request.POST, request.FILES)
        if form.is_valid():
            prediction = form.save(commit=False)
            prediction.user = request.user

            # Model tahmin işlemi
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = ResNetLungCancer(num_classes=4)
            model.load_state_dict(torch.load('lung_cancer_model.pth', map_location=device))
            model = model.to(device)
            model.eval()

            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            image = Image.open(prediction.image).convert('RGB')
            input_tensor = preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                predicted_class = torch.argmax(output, dim=1).item()
                confidence = probabilities[0][predicted_class].item()

            class_names = ['Adenocarcinoma', 'Large Cell Carcinoma', 'Squamous Cell Carcinoma','Normal']
            prediction.prediction_result = class_names[predicted_class]
            prediction.confidence_score = confidence * 100
            prediction.save()

            prediction_result = prediction.prediction_result  # Tahmin sonucunu context'e ekle


            messages.success(request, 'Tahmin başarıyla gerçekleştirildi!')
    else:
        form = PredictionForm()
    return render(request, 'users-panel/dashboard.html', {'form': form, 'prediction_result': prediction_result})


@login_required
def prediction_history(request):
    predictions = Prediction.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'users-panel/prediction_history.html', {'predictions': predictions})

@login_required
def prediction_detail(request, prediction_id):
    prediction = get_object_or_404(Prediction, id=prediction_id, user=request.user)
    return render(request, 'users-panel//prediction_detail.html', {'prediction': prediction})


@login_required
def download_prediction(request, prediction_id):
    prediction = get_object_or_404(Prediction, id=prediction_id, user=request.user)

    # CSV için UTF-8 BOM ekle
    response = HttpResponse(content_type='text/csv; charset=utf-8-sig')
    response['Content-Disposition'] = f'attachment; filename="tahmin_{prediction_id}.csv"'

    writer = csv.writer(response)

    # Kişisel Bilgiler
    writer.writerow(["--- KİŞİSEL BİLGİLER ---", ""])
    writer.writerow(["Tarih", prediction.created_at.strftime("%d/%m/%Y %H:%M")])
    writer.writerow(["Yaş", prediction.age])
    writer.writerow(["Cinsiyet", prediction.get_gender_display()])
    writer.writerow(["Konum", prediction.location])

    # Sigara Bilgileri
    writer.writerow(["", ""])
    writer.writerow(["--- SİGARA BİLGİLERİ ---", ""])
    writer.writerow(["Sigara Kullanımı", "Evet" if prediction.smoking_history else "Hayır"])
    if prediction.smoking_history:
        writer.writerow(["Sigara Kullanım Yılı", prediction.smoking_years or "Belirtilmemiş"])
        writer.writerow(["Günlük Sigara Sayısı", prediction.cigarettes_per_day or "Belirtilmemiş"])
        writer.writerow(["Sigarayı Bırakma Tarihi", prediction.quit_smoking_date.strftime(
            "%d/%m/%Y") if prediction.quit_smoking_date else "Belirtilmemiş"])
    writer.writerow(["Pasif İçicilik", "Var" if prediction.passive_smoking else "Yok"])

    # Tıbbi Geçmiş
    writer.writerow(["", ""])
    writer.writerow(["--- TIBBİ GEÇMİŞ ---", ""])
    writer.writerow(["Akciğer Hastalığı", "Var" if prediction.has_lung_disease else "Yok"])
    if prediction.has_lung_disease:
        writer.writerow(["Akciğer Hastalığı Detayları", prediction.lung_diseases or "Belirtilmemiş"])
    writer.writerow(["Geçirilmiş Akciğer Enfeksiyonu", "Var" if prediction.previous_lung_infections else "Yok"])
    writer.writerow(["Kronik Öksürük", "Var" if prediction.chronic_cough else "Yok"])
    if prediction.chronic_cough:
        writer.writerow(["Öksürük Süresi (Ay)", prediction.cough_duration or "Belirtilmemiş"])

    # Risk Faktörleri
    writer.writerow(["", ""])
    writer.writerow(["--- RİSK FAKTÖRLERİ ---", ""])
    writer.writerow(["Mesleki Maruziyet", "Var" if prediction.occupational_exposure else "Yok"])
    if prediction.occupational_exposure:
        writer.writerow(["Maruziyet Detayları", prediction.exposure_details or "Belirtilmemiş"])
    writer.writerow(["Hava Kirliliği Maruziyeti", "Var" if prediction.air_pollution_exposure else "Yok"])

    # Semptomlar
    writer.writerow(["", ""])
    writer.writerow(["--- SEMPTOMLAR ---", ""])
    writer.writerow(["Nefes Darlığı", "Var" if prediction.shortness_of_breath else "Yok"])
    writer.writerow(["Göğüs Ağrısı", "Var" if prediction.chest_pain else "Yok"])
    writer.writerow(["Kilo Kaybı", "Var" if prediction.weight_loss else "Yok"])
    writer.writerow(["Yorgunluk", "Var" if prediction.fatigue else "Yok"])
    writer.writerow(["Balgam Üretimi", "Var" if prediction.sputum_production else "Yok"])

    # Tahmin Sonuçları
    writer.writerow(["", ""])
    writer.writerow(["--- TAHMİN SONUÇLARI ---", ""])
    writer.writerow(["Sonuç", prediction.prediction_result])
    writer.writerow(["Güven Skoru", f"{prediction.confidence_score:.2f}%"])

    # Görüntüyü indirmek için ZIP dosyası oluştur
    if prediction.image:
        buffer = BytesIO()
        with ZipFile(buffer, 'w') as zip_file:
            # CSV dosyasını ZIP'e ekle
            zip_file.writestr('tahmin_raporu.csv', response.content)

            # Görüntüyü ZIP'e ekle
            image_name = os.path.basename(prediction.image.name)
            zip_file.write(prediction.image.path, image_name)

        # ZIP dosyasını döndür
        buffer.seek(0)
        response = HttpResponse(buffer.read(), content_type='application/zip')
        response['Content-Disposition'] = f'attachment; filename="tahmin_{prediction_id}_rapor.zip"'

    return response


@login_required
def export_predictions(request):
    predictions = Prediction.objects.filter(user=request.user)

    # ZIP dosyasını oluştur
    zip_filename = f"predictions_{request.user.username}.zip"
    zip_path = os.path.join("media", "exports", zip_filename)
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)

    with zipfile.ZipFile(zip_path, 'w') as zip_file:
        # CSV Dosyasını oluştur
        csv_filename = "predictions.csv"
        csv_path = os.path.join("media", "exports", csv_filename)

        with open(csv_path, "w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Tarih", "Yaş", "Cinsiyet", "Sonuç", "Güven Skoru", "Görüntü"])

            for prediction in predictions:
                writer.writerow([
                    prediction.created_at.strftime("%Y-%m-%d %H:%M"),
                    prediction.age,
                    prediction.get_gender_display(),
                    prediction.prediction_result,
                    f"{prediction.confidence_score:.2f}%",
                    os.path.basename(prediction.image.name) if prediction.image else "Yok"
                ])

                # Görüntüyü ZIP dosyasına ekle
                if prediction.image:
                    image_path = prediction.image.path
                    zip_file.write(image_path, os.path.basename(image_path))

        # CSV'yi ZIP dosyasına ekle
        zip_file.write(csv_path, os.path.basename(csv_path))

    # ZIP dosyasını yanıt olarak döndür
    with open(zip_path, "rb") as zip_file:
        response = HttpResponse(zip_file.read(), content_type="application/zip")
        response["Content-Disposition"] = f'attachment; filename="{zip_filename}"'

    return response


@login_required
def treatment_recommendation(request, prediction_id):
    prediction = get_object_or_404(Prediction, id=prediction_id, user=request.user)

    patient_info = {
        'yaş': prediction.age,
        'cinsiyet': 'Erkek' if prediction.gender == 'M' else 'Kadın',
        'tanı': prediction.prediction_result,
        'güven_skoru': f"%{prediction.confidence_score * 1:.2f}",
    }

    context = {
        'prediction': prediction,
        'patient_info': patient_info,
    }

    return render(request, 'treatment_recommendation.html', context)


@login_required
def get_recommendation(request, prediction_id):
    prediction = get_object_or_404(Prediction, id=prediction_id, user=request.user)

    # Gemini model konfigürasyonu
    genai.configure(api_key=settings.GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Hasta bilgilerini toplama
    patient_info = {
        'yaş': prediction.age,
        'cinsiyet': 'Erkek' if prediction.gender == 'M' else 'Kadın',
        'tanı': prediction.prediction_result,
        'güven_skoru': f"%{prediction.confidence_score * 1:.2f}",
    }

    # Sigara kullanım bilgileri
    smoking_info = []
    if prediction.smoking_history:
        smoking_info.append(f"- {prediction.smoking_years} yıldır sigara kullanıyor")
        smoking_info.append(f"- Günde {prediction.cigarettes_per_day} adet sigara")
        if prediction.quit_smoking_date:
            smoking_info.append(f"- Sigarayı bırakma tarihi: {prediction.quit_smoking_date}")
    if prediction.passive_smoking:
        smoking_info.append("- Pasif içici")

    # Semptomlar
    symptoms = []
    if prediction.shortness_of_breath:
        symptoms.append("nefes darlığı")
    if prediction.chest_pain:
        symptoms.append("göğüs ağrısı")
    if prediction.weight_loss:
        symptoms.append("kilo kaybı")
    if prediction.fatigue:
        symptoms.append("yorgunluk")
    if prediction.sputum_production:
        symptoms.append("balgam üretimi")

    # Risk faktörleri
    risk_factors = []
    if prediction.occupational_exposure:
        risk_factors.append(f"mesleki maruziyet: {prediction.exposure_details}")
    if prediction.air_pollution_exposure:
        risk_factors.append("hava kirliliği maruziyeti")
    if prediction.has_lung_disease:
        risk_factors.append(f"akciğer hastalıkları: {prediction.lung_diseases}")

    # Prompt oluşturma
    prompt = f"""
       Sen uzman bir göğüs hastalıkları doktoru olarak görev yapıyorsun. Aşağıda detayları verilen hasta için
       kapsamlı bir tedavi önerisi, hastalık seyri ve yaşam tarzı değişiklikleri hakkında profesyonel bir rapor hazırlamalısın.

       HASTA BİLGİLERİ:
       - Yaş: {patient_info['yaş']}
       - Cinsiyet: {patient_info['cinsiyet']}
       - Yapay Zeka Destekli Tanı: {patient_info['tanı']} (Güven Skoru: {patient_info['güven_skoru']})

       {"SİGARA KULLANIMI:" if smoking_info else ""}
       {chr(10).join(smoking_info) if smoking_info else ""}

       {"SEMPTOMLAR:" if symptoms else ""}
       {", ".join(symptoms) if symptoms else ""}

       {"RİSK FAKTÖRLERİ:" if risk_factors else ""}
       {chr(10).join(risk_factors) if risk_factors else ""}

       Lütfen aşağıdaki başlıklar altında detaylı bir rapor hazırla:
       1. Hastaya özel tedavi seçenekleri
       2. Benzer hastaların semptomları ve belirtileri
       3. Hastalığın tanı yöntemleri
       4. Hastalığın ilerleyişi ve takip süreci
       5. Önerilen yaşam tarzı değişiklikleri

       Her başlık için bilimsel ve güncel kaynaklara dayanan, ancak hasta için anlaşılır bir dil kullan.
       """

    try:
        # Gemini generation konfigürasyonu
        generation_config = {
            "temperature": 0.7,  # 0.0 - 1.0 arası, düşük değer daha tutarlı çıktılar
            "top_p": 0.8,  # 0.0 - 1.0 arası, düşük değer daha odaklı çıktılar
            "top_k": 40,  # Seçilecek en iyi token sayısı
            "max_output_tokens": 2048,  # Maksimum çıktı uzunluğu
            "candidate_count": 1,  # Üretilecek alternatif sayısı
        }

        # Güvenlik ayarları
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
        ]

        # Gemini'den yanıt alma
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        return JsonResponse({
            'success': True,
            'recommendation': response.text
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)



from django.views.decorators.csrf import csrf_exempt
import json

# Logger yapılandırması
logger = logging.getLogger(__name__)


class ChatbotConfig:
    """Chatbot için yapılandırma sınıfı"""

    TEMPERATURE = 0.7
    TOP_P = 0.8
    TOP_K = 40
    MAX_OUTPUT_TOKENS = 1026

    SYSTEM_PROMPT = """Sen deneyimli bir göğüs hastalıkları uzmanı ve akciğer kanseri konusunda uzmanlaşmış bir doktorsun. 
    Görevin hastalara ve yakınlarına akciğer kanseri hakkında doğru, güncel ve bilimsel bilgiler vermek.

    Yaklaşımın şu şekilde olmalı:
    1. Empatik ve profesyonel bir dil kullan
    2. Bilimsel ve güncel bilgiler ver
    3. Karmaşık tıbbi terimleri basitçe açıkla
    4. Umut verici ol ama gerçekçi ol
    5. Gerektiğinde doktora başvurmalarını öner

    Sadece şu konularda yanıt ver:
    - Akciğer kanseri belirtileri ve erken teşhis
    - Risk faktörleri ve korunma yöntemleri
    - Tanı yöntemleri ve aşamaları
    - Tedavi seçenekleri ve yaklaşımlar
    - Yaşam kalitesini artırma ve başa çıkma yöntemleri
    - Destek tedavileri ve rehabilitasyon
    - Takip süreci ve kontroller

    Bunların dışındaki konularda:
    "Üzgünüm, bu konu uzmanlık alanım dışında. Size en doğru bilgiyi verebilmesi için lütfen ilgili branş doktoruna danışın."

    Önemli: Asla kesin tanı koyma, ilaç önerme veya tedavi planı sunma. Her zaman bir sağlık kuruluşuna başvurmalarını öner.

    Kullanıcının sorusu: {user_message}
    """


class GeminiManager:
    """Gemini API etkileşimleri için yönetici sınıf"""

    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def get_generation_config(self) -> Dict[str, Any]:
        """Gemini generation yapılandırmasını döndürür"""
        return {
            "temperature": ChatbotConfig.TEMPERATURE,
            "top_p": ChatbotConfig.TOP_P,
            "top_k": ChatbotConfig.TOP_K,
            "max_output_tokens": ChatbotConfig.MAX_OUTPUT_TOKENS,
        }

    def generate_response(self, prompt: str) -> Optional[str]:
        """Gemini'den yanıt üretir"""
        try:
            generation_config = self.get_generation_config()
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            return response.text if response.text else None
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            return None


class ChatMessageHandler:
    """Sohbet mesajlarını işleyen sınıf"""

    def __init__(self):
        self.gemini_manager = GeminiManager()

    @staticmethod
    def validate_message(message: str) -> bool:
        """Kullanıcı mesajını doğrular"""
        if not message or not isinstance(message, str):
            return False
        if len(message.strip()) < 2:
            return False
        return True

    @staticmethod
    def sanitize_message(message: str) -> str:
        """Kullanıcı mesajını temizler"""
        return message.strip()

    def process_message(self, message: str) -> Dict[str, Any]:
        """Mesajı işler ve yanıt üretir"""
        try:
            if not self.validate_message(message):
                return {
                    'status': 'error',
                    'message': 'Geçersiz mesaj formatı'
                }

            sanitized_message = self.sanitize_message(message)
            prompt = ChatbotConfig.SYSTEM_PROMPT.format(user_message=sanitized_message)

            response = self.gemini_manager.generate_response(prompt)
            if not response:
                return {
                    'status': 'error',
                    'message': 'Yanıt üretilirken bir hata oluştu'
                }

            return {
                'status': 'success',
                'response': response
            }

        except Exception as e:
            logger.error(f"Message processing error: {str(e)}")
            return {
                'status': 'error',
                'message': 'İşlem sırasında bir hata oluştu'
            }


class ChatbotView(LoginRequiredMixin, TemplateView):
    login_url = '/login/'  # Kullanıcı giriş yapmamışsa yönlendirilecek URL
    redirect_field_name = 'next'
    template_name = 'chat.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # İhtiyaç duyulan ek context verileri buraya eklenebilir
        return context


@method_decorator(csrf_exempt, name='dispatch')
class ChatMessageView(LoginRequiredMixin, TemplateView):
    login_url = '/login/'
    redirect_field_name = 'next'
    """Chat mesajlarını işleyen view"""

    def post(self, request, *args, **kwargs):
        try:
            data = json.loads(request.body)
            user_message = data.get('message', '')

            handler = ChatMessageHandler()
            result = handler.process_message(user_message)

            if result['status'] == 'success':
                return JsonResponse(result)
            else:
                return JsonResponse(result, status=400)

        except json.JSONDecodeError:
            return JsonResponse({
                'status': 'error',
                'message': 'Geçersiz JSON formatı'
            }, status=400)

        except Exception as e:
            logger.error(f"Unexpected error in ChatMessageView: {str(e)}")
            return JsonResponse({
                'status': 'error',
                'message': 'Beklenmeyen bir hata oluştu'
            }, status=500)
