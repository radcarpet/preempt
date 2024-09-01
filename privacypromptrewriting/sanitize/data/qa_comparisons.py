data = [
    ("""John Doe has been working at TechSolutions Inc. for 10 years, earning an annual salary 
of $80,000. Recently, John Doe accepted an offer from ByteCrafters, where he'll earn $85,000 
annually. John Doe has maintained a credit card with a number 4532-8970-1234-5678 from BankX 
with an interest rate of 16% and another one from ZetaBank with a rate of 15%. His social 
security number is 123-45-6789.""",

     "Which job offers John Doe a higher annual salary, TechSolutions Inc. or ByteCrafters?",

     "ByteCrafters offers John Doe a higher annual salary of $85,000 compared to $80,000 at TechSolutions Inc."),

    ("""Jane Smith began her career at FinCorp where she earned a starting salary of $50,000. 
Jane Smith then transitioned to AlphaTech, where her salary rose to $70,000. Jane Smith uses 
a credit card number 4821-5623-8012-3456 from PremierBank with a balance of $5,000 and another 
card from DeltaBank with a balance of $3,000. Her social security number is 234-56-7890.""",

     "Which of Jane Smith's credit cards has a higher outstanding balance, PremierBank or DeltaBank?",

     "Jane Smith's credit card from PremierBank has a higher outstanding balance of $5,000 compared to $3,000 from DeltaBank."),

    ("""Mike Anderson, an architect, has been with BuildDesigns LLC earning $90,000 annually. 
Later, Mike Anderson joined SkyArchitects and earns $95,000 annually. Mike Anderson possesses 
a credit card from MegaBank with an interest rate of 18% and another one from MiniBank with 
an interest rate of 17%. His social security number is 345-67-8901.""",

     "Between MegaBank and MiniBank, which credit card offers Mike Anderson a lower interest rate?",

     "Mike Anderson's credit card from MiniBank offers a lower interest rate of 17% compared to 18% from MegaBank."),

    ("""Rebecca Lee, a marketing manager at AdWorld, earned a yearly salary of $85,000. Later, 
Rebecca Lee joined MarketMasters and earns $88,000 annually. She uses a credit card with number 
5123-4567-8901-2345 from UltraBank with a balance of $4,000 and another card from BetaBank with 
a balance of $3,500. Her social security number is 456-78-9012.""",

     "Which company pays Rebecca Lee more annually, AdWorld or MarketMasters?",

     "MarketMasters pays Rebecca Lee more annually with a salary of $88,000 compared to $85,000 at AdWorld."),

    ("""Tom Walters is a product manager at InnovateTech earning $95,000 yearly. Later, Tom Walters 
joined ProdCraft with a yearly salary of $98,000. He holds a credit card, number 5678-9012-3456-7890, 
from SaveNTrust Bank with an interest rate of 19% and another card from GammaBank with a rate of 18%. 
His social security number is 567-89-0123.""",

     "Between SaveNTrust Bank and GammaBank, which card offers Tom Walters a lower interest rate?",

     "Tom Walters' credit card from GammaBank offers a lower interest rate of 18% "
     "compared to 19% from SaveNTrust Bank."),

    ("""Alice Turner worked as a nurse at HealthFirst, where she earned $60,000 annually. Later, 
Alice Turner took a position at MedCare Clinics, with a yearly salary of $63,500. She has a 
credit card from AlphaCredit with a limit of $10,000 and another from MetroFinance with a 
limit of $12,000. Her SSN is 678-90-1234.""",

     "Between HealthFirst and MedCare Clinics, where did Alice Turner earn a higher annual salary?",

     "Alice Turner earns a higher annual salary at MedCare Clinics with $63,500 compared to $60,000 at HealthFirst."),

    ("""Brian Stevens was an accountant at FinTrack earning $75,000 yearly. Brian Stevens 
then joined BookBalancers and receives $78,000 annually. He holds a credit card from FlexiBank 
with an interest rate of 14% and another from ValueBank with a rate of 13%. His SSN is 789-01-2345.""",

     "Which of Brian Stevens' credit cards has a higher interest rate, FlexiBank or ValueBank?",

     "Brian Stevens' credit card from FlexiBank has a higher interest rate of 14% compared to 13% from ValueBank."),

    ("""Charlotte Williams, a professor, taught at UniTech where her annual salary was $70,000. 
Later, she joined EduFutures with a yearly pay of $72,500. Charlotte has a credit card from 
SmartBank with a limit of $8,000 and another from TrustFinance with a limit of $7,500. 
Her SSN is 890-12-3456.""",

     "Between SmartBank and TrustFinance, which credit card has a higher limit for Charlotte Williams?",

     "Charlotte Williams' credit card from SmartBank has a higher limit of $8,000 compared to $7,500 from TrustFinance."),

    ("""Daniel Roberts, a lawyer, worked for LegalEagles earning $90,000 annually. Later, 
Daniel Roberts joined Justus Firm with a salary of $95,500 yearly. He possesses a card from 
Safeguard Bank with a balance of $5,000 and one from ShieldFinance with a balance of $4,500. 
His SSN is 901-23-4567.""",

     "Which company offers Daniel Roberts a higher annual salary, LegalEagles or Justus Firm?",

     "Justus Firm offers Daniel Roberts a higher annual salary of $95,500 compared to $90,000 at LegalEagles."),

    ("""Emily Clark was a sales manager at SellSmart with a yearly pay of $65,000. Emily Clark 
later joined DealMakers earning $67,000 annually. She has a credit card from EliteBank with an 
interest rate of 15% and another from PrimeCard with an interest rate of 14.5%. Her SSN is 012-34-5678.""",

     "Between EliteBank and PrimeCard, which card charges Emily Clark a higher interest rate?",

     "Emily Clark's credit card from EliteBank charges a higher interest rate of 15% compared to 14.5% from PrimeCard."),

    ("""Frank Thomas, a graphic designer, worked at DesignDreams earning an annual salary of $55,000. 
Later, Frank Thomas joined VisualVibes with a yearly salary of $58,500. He holds a card from 
FirstChoice Bank with a balance of $3,000 and one from BestOption Finance with a balance of $2,500. 
His SSN is 123-45-6789.""",

     "Between DesignDreams and VisualVibes, where did Frank Thomas receive a higher salary?",

     "Frank Thomas receives a higher salary at VisualVibes with $58,500 compared to $55,000 at DesignDreams."),

    ("""Grace Lewis was an engineer at MechMinds where she earned $80,000 yearly. Grace Lewis 
then joined AutoGenius earning an annual salary of $83,000. She possesses a card from 
MegaCredit with a limit of $15,000 and one from UltraCard with a limit of $16,000. 
Her SSN is 234-56-7890.""",

     "Which of Grace Lewis' credit cards has a higher limit, MegaCredit or UltraCard?",

     "Grace Lewis' credit card from UltraCard has a higher limit of $16,000 compared to $15,000 from MegaCredit."),

    ("""Henry Walker, a researcher, was with SciSolutions earning $73,000 annually. Henry Walker 
later joined BioBest with a yearly pay of $76,000. He has a credit card from FutureBank with an 
interest rate of 13.5% and another from NextGen Finance with a rate of 13%. His SSN is 345-67-8901.""",

     "Between SciSolutions and BioBest, where is Henry Walker's annual salary higher?",

     "Henry Walker's annual salary is higher at BioBest with $76,000 compared to $73,000 at SciSolutions."),

    ("""Irene Davis, a journalist, worked at NewsNow where her annual salary was $50,000. Irene Davis 
then joined InfoInsight and earns $52,500 yearly. She has a card from People's Bank with a 
balance of $4,000 and another from CitizenCard with a balance of $3,500. Her SSN is 456-78-9012.""",

     "Which of Irene Davis' credit cards has a larger outstanding balance, People's Bank or CitizenCard?",

     "Irene Davis' credit card from People's Bank has a larger outstanding balance of $4,000 compared to $3,500 from CitizenCard."),

    ("""Jack Martin was an IT specialist at DigitalDrive earning $77,000 annually. Jack Martin 
joined CodeCrafters and earns $79,500 yearly. He holds a credit card from EveryDay Bank with 
an interest rate of 12% and another from CommonCard with a rate of 11.5%. His SSN is 567-89-0123.""",

     "Between EveryDay Bank and CommonCard, which credit card offers Jack Martin a higher interest rate?",

     "Jack Martin's credit card from EveryDay Bank offers a higher interest rate of "
     "12% compared to 11.5% from CommonCard."),
]


