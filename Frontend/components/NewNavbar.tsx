import CardNav from './cardnav';
import logo from '../../public/LOGO.png';

const NewNavbar = () => {
  const items = [
    {
      label: "Home",
      bgColor: "#f0efe7",
      textColor: "#0D0716",
      links: [
        { label: "Home", href: "/", ariaLabel: "Home Page" }
      ]
    },
    {
      label: "Dashboard",
      bgColor: "#f0efe7",
      textColor: "#0D0716",
      links: [
        { label: "Dashboard", href: "/dashboard", ariaLabel: "Dashboard Page" }
      ]
    },
    {
      label: "Chat",
      bgColor: "#f0efe7",
      textColor: "#0D0716",
      links: [
        { label: "Chat", href: "/chat", ariaLabel: "Chat Page" }
      ]
    },
    {
      label: "Globe",
      bgColor: "#f0efe7",
      textColor: "#0D0716",
      links: [
        { label: "Globe", href: "/globe", ariaLabel: "Globe Page" }
      ]
    },
    {
      label: "Upload Data",
      bgColor: "#f0efe7",
      textColor: "#0D0716",
      links: [
        { label: "Upload Data", href: "/upload", ariaLabel: "Upload Data Page" }
      ]
    },
    {
      label: "Float Types",
      bgColor: "#f0efe7",
      textColor: "#0D0716",
      links: [
        { label: "Float Types", href: "/float-types", ariaLabel: "Float Types Page" }
      ]
    },
    {
      label: "Videos",
      bgColor: "#f0efe7",
      textColor: "#0D0716",
      links: [
        { label: "Videos", href: "/videos", ariaLabel: "Videos Page" }
      ]
    },
    {
      label: "About",
      bgColor: "#f0efe7",
      textColor: "#0D0716",
      links: [
        { label: "About", href: "/about", ariaLabel: "About Page" }
      ]
    },
    {
      label: "Contact",
      bgColor: "#f0efe7",
      textColor: "#0D0716",
      links: [
        { label: "Contact", href: "/contact", ariaLabel: "Contact Page" }
      ]
    }
  ];


  return (
    <CardNav
      logo={logo.src}
      logoAlt="Company Logo"
      items={items}
      baseColor="#000"
      menuColor="#fff"
      buttonBgColor="#111"
      buttonTextColor="#fff"
      ease="elastic.out(1, 0.75)"
    />
  );
};

export default NewNavbar;
