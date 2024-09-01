data = [
    ("""Alice Turner worked as a nurse at HealthFirst, where she earned $60,000 annually. Later, 
Alice Turner took a position at MedCare Clinics, with a yearly salary of $63,500. She has a 
credit card from AlphaCredit with a limit of $10,000 and another from MetroFinance with a 
limit of $12,000. Her SSN is 678-90-1234.""",

     "Which employer pays Alice Turner more? 1. HealthFirst 2. MedCare Clinics",

     "2"),

    ("""Brian Stevens was an accountant at FinTrack earning $75,000 yearly. Brian Stevens 
then joined BookBalancers and receives $78,000 annually. He holds a credit card from FlexiBank 
with an interest rate of 14% and another from ValueBank with a rate of 13%. His SSN is 789-01-2345.""",

     "Which of Brian Stevens' credit cards has a higher interest rate? 1. FlexiBank 2. ValueBank",

     "1"),

    ("""Charlotte Williams, a professor, taught at UniTech where her annual salary was $70,000. 
Later, she joined EduFutures with a yearly pay of $72,500. Charlotte has a credit card from 
SmartBank with a limit of $8,000 and another from TrustFinance with a limit of $7,500. 
Her SSN is 890-12-3456.""",

     "Which credit card has a higher limit for Charlotte Williams? 1. SmartBank 2. TrustFinance",

     "1"),

    ("""Daniel Roberts, a lawyer, worked for LegalEagles earning $90,000 annually. Later, 
Daniel Roberts joined Justus Firm with a salary of $95,500 yearly. He possesses a card from 
Safeguard Bank with a balance of $5,000 and one from ShieldFinance with a balance of $4,500. 
His SSN is 901-23-4567.""",

     "Where does Daniel Roberts earn a higher salary? 1. LegalEagles 2. Justus Firm",

     "2"),

    ("""Emily Clark was a sales manager at SellSmart with a yearly pay of $65,000. Emily Clark 
later joined DealMakers earning $67,000 annually. She has a credit card from EliteBank with an 
interest rate of 15% and another from PrimeCard with an interest rate of 14.5%. Her SSN is 012-34-5678.""",

     "Which card charges Emily Clark a higher interest rate? 1. EliteBank 2. PrimeCard",

     "1"),

    ("""Frank Thomas, a graphic designer, worked at DesignDreams earning an annual salary of $55,000. 
Later, Frank Thomas joined VisualVibes with a yearly salary of $58,500. He holds a card from 
FirstChoice Bank with a balance of $3,000 and one from BestOption Finance with a balance of $2,500. 
His SSN is 123-45-6789.""",

     "Where did Frank Thomas receive a higher salary? 1. DesignDreams 2. VisualVibes",

     "2"),

    ("""Grace Lewis was an engineer at MechMinds where she earned $80,000 yearly. Grace Lewis 
then joined AutoGenius earning an annual salary of $83,000. She possesses a card from 
MegaCredit with a limit of $15,000 and one from UltraCard with a limit of $16,000. 
Her SSN is 234-56-7890.""",

     "Which of Grace Lewis' credit cards has a higher limit? 1. MegaCredit 2. UltraCard",

     "2"),

    ("""Henry Walker, a researcher, was with SciSolutions earning $73,000 annually. Henry Walker 
later joined BioBest with a yearly pay of $76,000. He has a credit card from FutureBank with an 
interest rate of 13.5% and another from NextGen Finance with a rate of 13%. His SSN is 345-67-8901.""",

     "Where is Henry Walker's annual salary higher? 1. SciSolutions 2. BioBest",

     "2"),

    ("""Irene Davis, a journalist, worked at NewsNow where her annual salary was $50,000. Irene Davis 
then joined InfoInsight and earns $52,500 yearly. She has a card from People's Bank with a 
balance of $4,000 and another from CitizenCard with a balance of $3,500. Her SSN is 456-78-9012.""",

     "Which of Irene Davis' credit cards has a larger outstanding balance? 1. People's Bank 2. CitizenCard",

     "1"),

    ("""Jack Martin was an IT specialist at DigitalDrive earning $77,000 annually. Jack Martin 
joined CodeCrafters and earns $79,500 yearly. He holds a credit card from EveryDay Bank with 
an interest rate of 12% and another from CommonCard with a rate of 11.5%. His SSN is 567-89-0123.""",

     "Which credit card offers Jack Martin a higher interest rate? 1. EveryDay Bank 2. CommonCard",

     "1"),

    ("""Katie Wilson, an event planner, worked at DreamEvents with an annual pay of $50,000. Later, 
Katie Wilson joined FantasyFests earning $52,000 yearly. She has a credit card from EventCard 
with a balance of $6,000 and another from FestFinance with a balance of $6,500. Her SSN is 678-90-1230.""",

     "Which employer pays Katie Wilson more? 1. DreamEvents 2. FantasyFests",

     "2"),

    ("""Liam Johnson was a real estate agent at EstateElite, earning $65,000 annually. Liam Johnson 
later joined RealtyRoyals and receives $67,500 yearly. He holds a credit card from HomeBank 
with a limit of $20,000 and another from PropertyCard with a limit of $21,500. His SSN is 789-01-2346.""",

     "Which credit card has a higher limit for Liam Johnson? 1. HomeBank 2. PropertyCard",

     "2"),

    ("""Megan Brown, a photographer, took pictures at PicturePerfect earning $40,000 yearly. 
Later, Megan Brown started at Snapshot Studios with a yearly pay of $42,000. She has a card from 
PhotoBank with an interest rate of 14% and another from ImageCard with a rate of 13.5%. Her SSN is 890-12-3457.""",

     "Which card charges Megan Brown a higher interest rate? 1. PhotoBank 2. ImageCard",

     "1"),

    ("""Nathan Evans, an architect, designed at ArchiArt earning $85,000 annually. Nathan Evans 
later joined DesignDynasty with a salary of $88,000 yearly. He possesses a card from BuildBank 
with a balance of $9,000 and one from DraftCard with a balance of $9,500. His SSN is 901-23-4568.""",

     "Which of Nathan Evans' credit cards has a larger outstanding balance? 1. BuildBank 2. DraftCard",

     "2"),

    ("""Olivia Smith, a fitness trainer, coached at FitForce with a yearly pay of $35,000. Olivia Smith 
later joined HealthHeros earning $37,500 annually. She has a credit card from VitalityBank 
with an interest rate of 16% and another from WellnessCard with an interest rate of 15.5%. Her SSN is 012-34-5679.""",

     "Which card charges Olivia Smith a higher interest rate? 1. VitalityBank 2. WellnessCard",

     "1"),

    ("""Paul Allen, a chef, cooked at GourmetGrill earning an annual salary of $45,000. Paul Allen 
later joined CulinaryKings with a yearly salary of $48,000. He holds a card from FoodieFinance 
with a limit of $13,000 and one from ChefCard with a limit of $14,000. His SSN is 123-45-6780.""",

     "Which of Paul Allen's credit cards has a higher limit? 1. FoodieFinance 2. ChefCard",

     "2"),

    ("""Quinn Adams, a fashion designer, worked at VogueVisions earning $70,000 yearly. Quinn Adams 
then joined GlamourGurus with an annual salary of $73,500. She possesses a card from TrendyBank 
with a balance of $11,000 and one from ChicCard with a balance of $10,500. Her SSN is 234-56-7891.""",

     "Which of Quinn Adams' credit cards has a larger outstanding balance? 1. TrendyBank 2. ChicCard",

     "1"),

    ("""Ryan Hughes, a musician, played at MelodyMakers earning $55,000 annually. Ryan Hughes 
later joined SymphonyStars with a yearly pay of $58,000. He has a credit card from TuneBank with an 
interest rate of 12.5% and another from RhythmCard with a rate of 12%. His SSN is 345-67-8902.""",

     "Which card charges Ryan Hughes a higher interest rate? 1. TuneBank 2. RhythmCard",

     "1"),

    ("""Sophia Turner, an artist, painted at ColorCanvas where her annual salary was $60,000. Sophia Turner 
then joined ArtisticAvenue and earns $63,000 yearly. She has a card from PaintBank with a 
limit of $9,000 and another from BrushCard with a limit of $9,500. Her SSN is 456-78-9013.""",

     "Which of Sophia Turner's credit cards has a higher limit? 1. PaintBank 2. BrushCard",

     "2"),

    ("""Tyler White, a software developer, coded at CodeCreators earning $95,000 annually. Tyler White 
joined SoftwareSages and earns $98,000 yearly. He holds a credit card from TechBank with 
an interest rate of 11% and another from DigitalCard with a rate of 10.5%. His SSN is 567-89-0124.""",

     "Which credit card offers Tyler White a higher interest rate? 1. TechBank 2. DigitalCard",

     "1"),
]


