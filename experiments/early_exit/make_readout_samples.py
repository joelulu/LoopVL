"""Create 10 synthetic images and 20 short/descriptive readout smoke questions."""
import argparse
import json
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New directory')
    args = parser.parse_args()
    from PIL import Image, ImageDraw, ImageFont
    args.output.mkdir(parents=True, exist_ok=False)
    font = ImageFont.load_default(size=24)
    rng = random.Random(42)
    samples = []
    for i in range(5):
        values = rng.sample(range(2, 9), 3)
        image = Image.new('RGB', (640, 400), 'white')
        draw = ImageDraw.Draw(image)
        for x, label, value in zip([100, 280, 460], 'ABC', values):
            top = 330-value*32
            draw.rectangle((x, top, x+80, 330), fill='steelblue')
            draw.text((x+25, 345), label, fill='black', font=font)
            draw.text((x+25, top-30), str(value), fill='black', font=font)
        name = f'chart_{i}.png'
        image.save(args.output/name)
        for style, prompt in [
            ('short', 'Which bar is the tallest? Answer with its label only.'),
            ('description', 'Describe this chart. State the value of each bar from left to right, then identify the tallest and shortest bars.')]:
            samples.append(dict(id=f'chart_{i}_{style}', image=name, prompt=prompt,
                                category=f'chart_{style}', expected_values=values))
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, color in enumerate(colors):
        count = i+2
        image = Image.new('RGB', (640, 400), 'white')
        draw = ImageDraw.Draw(image)
        for j in range(count):
            x, y = 60+(j%3)*200, 55+(j//3)*180
            draw.rectangle((x, y, x+90, y+90), fill=color)
        name = f'count_{i}.png'
        image.save(args.output/name)
        for style, prompt in [
            ('short', f'How many {color} squares are in the image? Answer with a number only.'),
            ('description', 'Describe the objects in this image, including their shape, color, number and arrangement.')]:
            samples.append(dict(id=f'count_{i}_{style}', image=name, prompt=prompt,
                                category=f'count_{style}', expected_count=count))
    (args.output/'samples.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in samples), encoding='utf-8')
    print(f'Created {len(samples)} questions on 10 unique synthetic images: {args.output / "samples.jsonl"}')


if __name__ == '__main__':
    main()
