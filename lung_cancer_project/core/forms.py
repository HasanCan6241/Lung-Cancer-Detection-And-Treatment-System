from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import ContactMessage, Prediction

class UserRegistrationForm(UserCreationForm):
    email = forms.EmailField()

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']

class ContactForm(forms.ModelForm):
    class Meta:
        model = ContactMessage
        fields = ['name', 'email', 'message']


class PredictionForm(forms.ModelForm):
    class Meta:
        model = Prediction
        fields = [
            'image', 'age', 'gender', 'location',
            'smoking_history', 'smoking_years', 'cigarettes_per_day',
            'passive_smoking', 'quit_smoking_date', 'family_history',
            'has_lung_disease', 'lung_diseases', 'previous_lung_infections',
            'chronic_cough', 'cough_duration', 'occupational_exposure',
            'exposure_details', 'air_pollution_exposure', 'shortness_of_breath',
            'chest_pain', 'weight_loss', 'fatigue', 'sputum_production'
        ]

        labels = {
            'image': 'BT Görüntüsü',
            'age': 'Yaş',
            'gender': 'Cinsiyet',
            'location': 'Yaşadığı Şehir',
            'smoking_history': 'Sigara Kullanım Geçmişi',
            'smoking_years': 'Kaç Yıldır Sigara Kullanıyorsunuz?',
            'cigarettes_per_day': 'Günlük Sigara Sayısı',
            'passive_smoking': 'Pasif İçicilik',
            'quit_smoking_date': 'Sigarayı Bırakma Tarihi',
            'family_history': 'Ailede Kanser Geçmişi',
            'has_lung_disease': 'Akciğer Hastalığı Geçmişi',
            'lung_diseases': 'Geçirdiğiniz Akciğer Hastalıkları (KOAH, Astım vb.)',
            'previous_lung_infections': 'Geçirilmiş Akciğer Enfeksiyonu',
            'chronic_cough': 'Kronik Öksürük',
            'cough_duration': 'Öksürük Süresi (Ay)',
            'occupational_exposure': 'Mesleki Maruziyet',
            'exposure_details': 'Maruziyet Detayları (Kimyasal, Toz vb.)',
            'air_pollution_exposure': 'Hava Kirliliği Maruziyeti',
            'shortness_of_breath': 'Nefes Darlığı',
            'chest_pain': 'Göğüs Ağrısı',
            'weight_loss': 'Kilo Kaybı',
            'fatigue': 'Yorgunluk',
            'sputum_production': 'Balgam Çıkarma'
        }

        widgets = {
            'age': forms.NumberInput(attrs={'class': 'form-control', 'min': '0', 'max': '120'}),
            'gender': forms.Select(attrs={'class': 'form-select'}),
            'location': forms.TextInput(attrs={'class': 'form-control'}),
            'smoking_years': forms.NumberInput(attrs={'class': 'form-control', 'min': '0'}),
            'cigarettes_per_day': forms.NumberInput(attrs={'class': 'form-control', 'min': '0'}),
            'quit_smoking_date': forms.DateInput(attrs={'class': 'form-control', 'type': 'date'}),
            'lung_diseases': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': '3',
                'placeholder': 'Örn: KOAH, Astım, Bronşit...'
            }),
            'cough_duration': forms.NumberInput(attrs={'class': 'form-control', 'min': '0'}),
            'exposure_details': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': '3',
                'placeholder': 'Örn: Kimyasal maruziyeti, tozlu ortam, asbest...'
            }),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['image'].widget.attrs.update({'class': 'form-control'})
        self.fields['smoking_years'].required = False
        self.fields['cigarettes_per_day'].required = False
        self.fields['quit_smoking_date'].required = False
        self.fields['lung_diseases'].required = False
        self.fields['cough_duration'].required = False
        self.fields['exposure_details'].required = False

    def clean_image(self):
        image = self.cleaned_data.get('image')
        if image:
            if image.size > 5 * 1024 * 1024:
                raise forms.ValidationError("Dosya boyutu 5MB'dan küçük olmalıdır.")
            allowed_types = ['image/jpeg', 'image/png', 'image/dicom']
            if image.content_type not in allowed_types:
                raise forms.ValidationError("Sadece JPEG, PNG ve DICOM dosyaları yüklenebilir.")
        return image