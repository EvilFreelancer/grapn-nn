import scrapy
import json
import re


class HabrCareerSpider(scrapy.Spider):
    name = 'habr_career'
    allowed_domains = ['career.habr.com']
    start_urls = ['https://career.habr.com/vacancies?page=1&type=all']

    def parse(self, response):
        for vacancy in response.css('div.section-group.section-group--gap-medium div.vacancy-card'):
            vacancy_url = vacancy.css('a.vacancy-card__title-link::attr(href)').get()
            yield response.follow(vacancy_url, self.parse_vacancy)

        next_page = response.url.split('page=')[1].split('&')[0]
        next_page_number = int(next_page) + 1
        next_page_url = f'https://career.habr.com/vacancies?page={next_page_number}&type=all'

        if next_page_number <= 143:  # 3550 вакансий / 25 вакансий на страницу = 142 страницы
            yield scrapy.Request(url=next_page_url, callback=self.parse)

    def parse_vacancy(self, response):
        # Get basic information
        title = response.css('h1.page-title__title::text').get().strip()
        description = response.css('div.vacancy-description__text ::text').getall()
        description = '\n'.join([s.replace('  ', ' ').strip() for s in description]).replace(' ', ' ').replace('​', '')
        requirements = response.css(
            'div.content-section > span.inline-list > span:not([class]) > span > a.link-comp ::text'
        ).getall()

        # Extract information from json-ld tag
        qualification = None
        specialization = None
        cities = []
        json_ld_script = response.xpath('//script[@type="application/ld+json"]/text()').get()
        if json_ld_script:
            vacancy_data = json.loads(json_ld_script)

            # Parse cities
            # if isinstance(job_locations, list):
            #     country = 'Российская Федерация'
            #     for location in job_locations:
            #         city = location.get('address')
            #         if city:
            #             cities.append(city)
            # elif isinstance(job_locations, dict):
            #     address = job_locations.get('address', {})
            #     address_country = address.get('addressCountry', {})
            #     country = address_country.get('name')
            #     city = address.get('addressLocality')
            #     if city:
            #         cities.append(city)

            # Parse description
            ld_description = vacancy_data.get('description', '')
            qualification_match = re.search(r'\. Квалификация: (\. |[^<.]+)', ld_description)
            specialization_match = re.search(r'\. Специализации:\s*([^<.]+)', ld_description)
            qualification = qualification_match.group(1) if qualification_match else ''
            specialization = specialization_match.group(1) if specialization_match else ''

        yield {
            'title': title,
            # 'cities': cities,
            'requirements': requirements,
            'qualification': qualification,
            'specialization': specialization,
            'description': description
        }
