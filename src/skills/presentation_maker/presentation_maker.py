import json
import os
from pathlib import Path
from langchain_core.tools import tool

@tool
def create_presentation_file(filename: str, title: str, slides: list) -> str:
    """
    Создает интерактивную HTML-презентацию на основе предоставленного контента.
    
    Args:
        filename: Имя файла (должно заканчиваться на .html).
        title: Главный заголовок презентации.
        slides: Список словарей, где каждый словарь - это слайд:
                {"title": "Заголовок слайда", "content": ["пункт 1", "пункт 2"] или "текст"}
    """
    try:
        if not filename.endswith('.html'):
            filename += '.html'
            
        html_template = """
        <!DOCTYPE html>
        <html lang="ru">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{main_title}</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; background: #f0f2f5; display: flex; justify-content: center; align-items: center; height: 100vh; overflow: hidden; }}
                .presentation-container {{ width: 80%; height: 80%; background: white; box-shadow: 0 10px 30px rgba(0,0,0,0.1); border-radius: 12px; position: relative; display: flex; flex-direction: column; }}
                .slide {{ display: none; padding: 60px; flex-grow: 1; animation: fadeIn 0.5s; }}
                .slide.active {{ display: block; }}
                h1 {{ color: #1a73e8; font-size: 3em; margin-bottom: 20px; }}
                h2 {{ color: #3c4043; font-size: 2.5em; border-bottom: 2px solid #1a73e8; padding-bottom: 10px; }}
                ul {{ font-size: 1.5em; line-height: 1.6; color: #5f6368; }}
                .controls {{ position: absolute; bottom: 20px; right: 20px; display: flex; gap: 10px; }}
                button {{ padding: 10px 20px; cursor: pointer; background: #1a73e8; color: white; border: none; border-radius: 5px; font-size: 1em; }}
                button:hover {{ background: #1557b0; }}
                .slide-number {{ position: absolute; bottom: 20px; left: 20px; color: #9aa0a6; font-size: 1.2em; }}
                @keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
            </style>
        </head>
        <body>
            <div class="presentation-container">
                {slides_html}
                <div class="slide-number" id="slideNum">1 / {total}</div>
                <div class="controls">
                    <button onclick="prevSlide()">Назад</button>
                    <button onclick="nextSlide()">Вперед</button>
                </div>
            </div>

            <script>
                let currentSlide = 0;
                const slides = document.querySelectorAll('.slide');
                const slideNum = document.getElementById('slideNum');

                function showSlide(n) {{
                    slides[currentSlide].classList.remove('active');
                    currentSlide = (n + slides.length) % slides.length;
                    slides[currentSlide].classList.add('active');
                    slideNum.innerText = (currentSlide + 1) + ' / ' + slides.length;
                }}

                function nextSlide() {{ showSlide(currentSlide + 1); }}
                function prevSlide() {{ showSlide(currentSlide - 1); }}
                
                document.addEventListener('keydown', (e) => {{
                    if (e.key === 'ArrowRight') nextSlide();
                    if (e.key === 'ArrowLeft') prevSlide();
                }});
            </script>
        </body>
        </html>
        """
        
        slides_html = ""
        # Титульный слайд
        slides_html += f'<div class="slide active"><h1>{title}</h1><p style="font-size:1.5em; color:#5f6368;">Презентация создана AI Агентом</p></div>'
        
        for slide in slides:
            content_html = ""
            content = slide.get('content', [])
            if isinstance(content, list):
                content_html = "<ul>" + "".join([f"<li>{item}</li>" for item in content]) + "</ul>"
            else:
                content_html = f"<p style='font-size:1.5em;'>{content}</p>"
                
            slides_html += f"""
            <div class="slide">
                <h2>{slide.get('title', 'Без названия')}</h2>
                {content_html}
            </div>
            """
            
        full_html = html_template.format(
            main_title=title,
            slides_html=slides_html,
            total=len(slides) + 1
        )
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(full_html)
            
        return f"Презентация успешно создана: {os.path.abspath(filename)}"
    except Exception as e:
        return f"Ошибка при создании презентации: {str(e)}"
