import './App.css';
import GoogleFontLoader from 'react-google-font-loader';
import SectionWrapper from './SectionWrapper';
import Header from './Header';
import SectionHero from './SectionHero';
import SectionDemo from './SectionDemo';
import SectionCaps from './SectionCaps';
import SectionTech from './SectionTech';
import SectionAbout from './SectionAbout';
import Footer from './Footer';

function App() {
	return (
		<>
			<GoogleFontLoader
				fonts={[
					{
						font: "Chivo",
						weights: [400, 700],
					},
					{
						font: "Rubik",
						weights: [400, 500, 700],
					}
				]}
			/>
			
			<div className="App" id="home">
				<Header />
				
				<main>
					{/* Title and image */}
					<SectionWrapper>
						<SectionHero />
					</SectionWrapper>
					
					{/* Capabilities */}
					<SectionWrapper id="caps">
						<SectionCaps />
					</SectionWrapper>
					
					{/* Demo */}
					<SectionWrapper id="demo">
						<SectionDemo />
					</SectionWrapper>
					
					{/* Technical details */}
					<SectionWrapper id="tech">
						<SectionTech />
					</SectionWrapper>
					
					{/* About us */}
					<SectionWrapper id="about">
						<SectionAbout />
					</SectionWrapper>
				</main>
				
				<Footer />
			</div>
		</>
	);
}

export default App;
