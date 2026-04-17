from django import forms
from .models import DesignRequest

class DesignForm(forms.ModelForm):

    SCENE_CHOICES = [
        ('interior', 'Interior'),
        ('exterior', 'Exterior'),
    ]

    scene_type = forms.ChoiceField(choices=SCENE_CHOICES)

    class Meta: 
        model = DesignRequest
        fields = ['original_image', 'prompt', 'scene_type']