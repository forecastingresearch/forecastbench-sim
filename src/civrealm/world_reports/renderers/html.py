"""HTML renderer for world reports"""

from typing import Dict, List
from pathlib import Path
from ..sections.base_section import SectionData


class HTMLRenderer:
    """Render report sections to HTML"""

    def __init__(self, output_dir: str, turn: int):
        """Initialize HTML renderer

        Args:
            output_dir: Directory to save output files
            turn: Turn number for this report
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.turn = turn
        self.images_dir = self.output_dir / f'turn_{turn:03d}_images'
        self.images_dir.mkdir(exist_ok=True)

    def render(self, sections: List[SectionData]) -> str:
        """Render complete HTML report

        Args:
            sections: List of SectionData to render

        Returns:
            Path to generated HTML file
        """
        # Save all images
        for section in sections:
            for img_name, img_buf in section.images.items():
                img_path = self.images_dir / f'{img_name}.png'
                img_buf.seek(0)
                with open(img_path, 'wb') as f:
                    f.write(img_buf.read())

        # Build HTML document
        html_parts = []
        html_parts.append(self._get_html_header())

        # Title page
        html_parts.append('<div class="title-page">')
        html_parts.append(f'<h1>World Report</h1>')
        html_parts.append(f'<h2>Turn {self.turn}</h2>')
        html_parts.append('<p class="subtitle">Comprehensive Analysis of World State</p>')
        html_parts.append('</div>')

        # Table of contents
        html_parts.append('<div class="toc">')
        html_parts.append('<h2>Table of Contents</h2>')
        html_parts.append('<ol>')
        for idx, section in enumerate(sections, 1):
            html_parts.append(f'<li><a href="#section{idx}">{section.title}</a></li>')
        html_parts.append('</ol>')
        html_parts.append('</div>')

        # Sections
        for idx, section in enumerate(sections, 1):
            html_parts.append(f'<div id="section{idx}" class="section">')

            # Replace image references with relative paths
            content = section.content
            for img_name in section.images.keys():
                content = content.replace(
                    f'{img_name}.png',
                    f'turn_{self.turn:03d}_images/{img_name}.png'
                )

            html_parts.append(content)
            html_parts.append('</div>')

        html_parts.append(self._get_html_footer())

        # Write HTML file
        html_file = self.output_dir / f'turn_{self.turn:03d}_report.html'
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html_parts))

        return str(html_file)

    def _get_html_header(self) -> str:
        """Get HTML document header with CSS"""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>World Report - Turn ''' + str(self.turn) + '''</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
            padding: 20px;
        }

        .title-page {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 80px 40px;
            text-align: center;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        .title-page h1 {
            font-size: 3em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }

        .title-page h2 {
            font-size: 2em;
            margin-bottom: 10px;
        }

        .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
        }

        .toc {
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .toc h2 {
            color: #667eea;
            margin-bottom: 20px;
        }

        .toc ol {
            margin-left: 30px;
        }

        .toc li {
            margin: 10px 0;
        }

        .toc a {
            color: #764ba2;
            text-decoration: none;
            font-size: 1.1em;
        }

        .toc a:hover {
            text-decoration: underline;
        }

        .section {
            background: white;
            padding: 40px;
            margin-bottom: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        h2 {
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }

        h3 {
            color: #764ba2;
            margin-top: 30px;
            margin-bottom: 15px;
        }

        h4 {
            color: #555;
            margin-top: 20px;
            margin-bottom: 10px;
        }

        p {
            margin: 10px 0;
        }

        .overview-info {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }

        .statistics {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }

        .statistics ul {
            list-style: none;
            margin-left: 0;
        }

        .statistics li {
            padding: 5px 0;
        }

        table.data-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }

        table.data-table caption {
            caption-side: top;
            padding: 10px;
            font-weight: bold;
            font-size: 1.1em;
            color: #667eea;
            text-align: left;
        }

        table.data-table th {
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }

        table.data-table td {
            padding: 10px 12px;
            border-bottom: 1px solid #e0e0e0;
        }

        table.data-table tr:hover {
            background: #f5f5f5;
        }

        table.data-table tr:last-child td {
            border-bottom: none;
        }

        .mini-maps-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }

        .mini-map {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }

        .mini-map img {
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .mini-map .caption {
            margin-top: 10px;
            font-size: 0.95em;
            color: #555;
        }

        .territory-maps {
            margin: 20px 0;
        }

        .territory-map {
            margin: 20px 0;
        }

        .territory-map img {
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        img {
            max-width: 100%;
            height: auto;
        }

        @media print {
            body {
                background: white;
                padding: 0;
            }

            .section {
                page-break-inside: avoid;
                box-shadow: none;
            }

            .title-page {
                page-break-after: always;
            }
        }
    </style>
</head>
<body>
'''

    def _get_html_footer(self) -> str:
        """Get HTML document footer"""
        return '''
</body>
</html>
'''
