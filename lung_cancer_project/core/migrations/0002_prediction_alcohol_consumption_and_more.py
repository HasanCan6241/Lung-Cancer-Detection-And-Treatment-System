# Generated by Django 5.1.5 on 2025-02-08 22:44

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='prediction',
            name='alcohol_consumption',
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name='prediction',
            name='genetic_testing',
            field=models.TextField(blank=True),
        ),
        migrations.AddField(
            model_name='prediction',
            name='medical_history',
            field=models.TextField(blank=True),
        ),
        migrations.AddField(
            model_name='prediction',
            name='physical_activity',
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name='prediction',
            name='pollution_exposure',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='prediction',
            name='profession',
            field=models.CharField(blank=True, max_length=100),
        ),
        migrations.AddField(
            model_name='prediction',
            name='symptoms',
            field=models.TextField(blank=True),
        ),
        migrations.AddField(
            model_name='prediction',
            name='test_results',
            field=models.TextField(blank=True),
        ),
        migrations.AlterField(
            model_name='prediction',
            name='smoking_history',
            field=models.IntegerField(choices=[(0, 'Hiç İçmedim'), (1, 'Eski İçici'), (2, 'Aktif İçici')], default=0),
        ),
    ]
